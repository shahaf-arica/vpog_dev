from vpog.models.vpog_model import VPOGModel
from vpog.models.flow_head import pack_valid_qt_pairs
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

class VPOGPredictor:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_correspondences_on_query(
        self,
        image_query: torch.Tensor,          # [B,3,H,W]
        template_images: torch.Tensor,      # [B,S,3,H,W]
        template_poses: torch.Tensor,       # [B,S,4,4]
        *,
        top_k_templates: Optional[int] = None,
        top_k_patches: Optional[int] = None,
        weight_thresh: float = 0.0,
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Returns:
          results: List length B.
            results[b] is a dict mapping template_index(int) -> template_dict
            The dict has exactly top_k_templates keys (or S keys if top_k_templates is None).

          For each template_dict:
            - "template_idx": int
            - "template_rank": int (0 is best among chosen)
            - "template_score": float (S_t)
            - "coarse": dict with:
                * "q_center_uv": FloatTensor [Nc,2]  (u,v on query)
                * "t_center_uv": FloatTensor [Nc,2]  (u,v on template)
                * "q_idx": LongTensor [Nc]
                * "t_idx": LongTensor [Nc]
                * "patch_score": FloatTensor [Nc]    (softmax-max confidence for the selected patch)
            - "refined": dict with:
                * "t_uv": FloatTensor [Nr,2]         (u,v on template)
                * "q_uv": FloatTensor [Nr,2]         (u,v on query, predicted)
                * "w": RockyTensor [Nr]              (dense_w)
                * "pair_row": LongTensor [Nr]        (index into the coarse rows within this template entry)
                  (lets you associate refined pixels back to which coarse patch-pair they came from)
        """
        # -----------------------------
        # Run model once
        # -----------------------------
        image_query = image_query.to(self.device)
        template_images = template_images.to(self.device)
        template_poses = template_poses.to(self.device)

        out = self.model(
            query_images=image_query,
            template_images=template_images,
            template_poses=template_poses,
        )
        q_tokens_aa = out["query_tokens_aa"]      # [B, Nq, C]
        t_tokens_aa = out["template_tokens_aa"]   # [B, S, Nt+1, C]

        B, Nq, _ = q_tokens_aa.shape
        _, S, Nt1, _ = t_tokens_aa.shape
        Nt = Nq  # by construction

        # derive patch grid side
        H_p = int(round(Nq ** 0.5))
        if H_p * H_p != Nq:
            raise ValueError(f"Expected Nq to be square. Got Nq={Nq} (sqrt={Nq**0.5}).")

        # Image tokens only (critical for correct t_idx indexing)
        t_img_aa = t_tokens_aa[:, :, :Nt, :].contiguous()  # [B,S,Nt,C]

        # patch size (training convention)
        ps = int(self.model.dense_flow_head.ps)

        # -----------------------------
        # Classification logits
        # -----------------------------
        logits = self.model.classification_head(
            q_tokens=q_tokens_aa,
            t_tokens=t_tokens_aa,   # include extra token => Nt1 classes
        )  # [B,S,Nq,Nt1]

        patch_cls_pred = logits.argmax(dim=-1)  # [B,S,Nq] in [0..Nt] (Nt=unseen)

        # Probabilities for scoring
        prob = F.softmax(logits, dim=-1)        # [B,S,Nq,Nt1]
        p_max, _ = prob.max(dim=-1)             # [B,S,Nq]
        valid_buddy = (patch_cls_pred >= 0) & (patch_cls_pred < Nt)  # [B,S,Nq]

        # per-patch score as in Co-op (max prob if matched else 0)
        score_patch = torch.where(valid_buddy, p_max, torch.zeros_like(p_max))  # [B,S,Nq]
        score_template = score_patch.sum(dim=-1)  # [B,S]

        # -----------------------------
        # Select top templates
        # -----------------------------
        if top_k_templates is None or top_k_templates >= S:
            Kt = S
            chosen_template_idx = torch.arange(S, device=self.device)[None, :].expand(B, S)  # [B,S]
            chosen_template_score = score_template  # [B,S]
        else:
            Kt = int(top_k_templates)
            chosen_template_score, chosen_template_idx = torch.topk(
                score_template, k=Kt, dim=1, largest=True, sorted=True
            )  # both [B,Kt]

        # template keep mask [B,S]
        template_keep = torch.zeros((B, S), dtype=torch.bool, device=self.device)
        template_keep.scatter_(1, chosen_template_idx, True)

        # -----------------------------
        # Select patches within chosen templates
        # -----------------------------
        keep_patch = template_keep[:, :, None] & valid_buddy  # [B,S,Nq]

        if top_k_patches is not None:
            Kp = int(top_k_patches)
            if Kp <= 0:
                raise ValueError("top_k_patches must be positive when not None.")

            neg_inf = torch.finfo(score_patch.dtype).min
            score_for_rank = torch.where(
                keep_patch, score_patch, torch.full_like(score_patch, neg_inf)
            )  # [B,S,Nq]

            kk = min(Kp, Nq)
            top_vals, top_idx = torch.topk(score_for_rank, k=kk, dim=2, largest=True, sorted=False)  # [B,S,kk]
            selected = top_vals > neg_inf / 2

            keep_patch_topk = torch.zeros((B, S, Nq), dtype=torch.bool, device=self.device)
            keep_patch_topk.scatter_(2, top_idx, selected)
            keep_patch = keep_patch_topk

        # build patch_cls_selected for packing
        patch_cls_selected = torch.full((B, S, Nq), -1, dtype=torch.long, device=self.device)
        patch_cls_selected[keep_patch] = patch_cls_pred[keep_patch]

        # pack only final pairs
        pack = pack_valid_qt_pairs(q_tokens_aa, t_img_aa, patch_cls_selected)
        b_idx = pack["b_idx"]  # [M]
        s_idx = pack["s_idx"]  # [M]
        q_idx = pack["q_idx"]  # [M]
        t_idx = pack["t_idx"]  # [M]
        M = int(pack["M"].item())

        # run dense head
        pred_flow, pred_b, dense_w = self.model.dense_flow_head.forward_packed(pack["q_tok"], pack["t_tok"])
        # pred_flow: [M,ps,ps,2]  t->q normalized by ps
        # dense_w:   [M,ps,ps]

        # -----------------------------
        # Compute coarse centers (q->t) for each packed pair m
        # training convention: center = idx*ps + ps//2, integer pixel coordinates
        # -----------------------------
        if M > 0:
            q_pi = torch.div(q_idx, H_p, rounding_mode="floor")
            q_pj = q_idx % H_p
            t_pi = torch.div(t_idx, H_p, rounding_mode="floor")
            t_pj = t_idx % H_p

            q_center_u = (q_pj * ps + (ps // 2)).to(torch.float32)
            q_center_v = (q_pi * ps + (ps // 2)).to(torch.float32)
            t_center_u = (t_pj * ps + (ps // 2)).to(torch.float32)
            t_center_v = (t_pi * ps + (ps // 2)).to(torch.float32)

            coarse_q_center_uv = torch.stack([q_center_u, q_center_v], dim=-1)  # [M,2]
            coarse_t_center_uv = torch.stack([t_center_u, t_center_v], dim=-1)  # [M,2]

            # per-pair patch_score (the same score used for ranking)
            patch_score_sel = score_patch[b_idx, s_idx, q_idx].to(torch.float32)  # [M]
        else:
            coarse_q_center_uv = q_tokens_aa.new_empty((0, 2), dtype=torch.float32)
            coarse_t_center_uv = q_tokens_aa.new_empty((0, 2), dtype=torch.float32)
            patch_score_sel = q_tokens_aa.new_empty((0,), dtype=torch.float32)

        # -----------------------------
        # Build refined pixel correspondences (t->q) as a FLAT LIST over surviving pixels
        # Output per pixel: (t_u,t_v) -> (q_u,q_v) plus weight
        # -----------------------------
        if M > 0:
            keep_pix = dense_w >= float(weight_thresh)  # [M,ps,ps]
            nz = keep_pix.nonzero(as_tuple=False)        # [K,3] with columns: [m, i, j]
            # if nothing survives, produce empty
            if nz.numel() == 0:
                pix_m = q_tokens_aa.new_empty((0,), dtype=torch.long)
                pix_i = q_tokens_aa.new_empty((0,), dtype=torch.long)
                pix_j = q_tokens_aa.new_empty((0,), dtype=torch.long)
                pix_w = q_tokens_aa.new_empty((0,), dtype=torch.float32)
                pix_t_uv = q_tokens_aa.new_empty((0, 2), dtype=torch.float32)
                pix_q_uv = q_tokens_aa.new_empty((0, 2), dtype=torch.float32)
            else:
                pix_m = nz[:, 0].long()
                pix_i = nz[:, 1].long()
                pix_j = nz[:, 2].long()

                pix_w = dense_w[pix_m, pix_i, pix_j].to(torch.float32)  # [K]

                # Template absolute pixel coords for buddy template patch
                t_pi = torch.div(t_idx[pix_m], H_p, rounding_mode="floor")  # [K]
                t_pj = t_idx[pix_m] % H_p                                   # [K]
                t_u = (t_pj * ps).to(torch.float32) + pix_j.to(torch.float32)
                t_v = (t_pi * ps).to(torch.float32) + pix_i.to(torch.float32)
                pix_t_uv = torch.stack([t_u, t_v], dim=-1)  # [K,2]

                # Query absolute pixel coords from predicted t->q flow:
                # q_uv = q_patch_center + pred_flow*ps
                q_pi = torch.div(q_idx[pix_m], H_p, rounding_mode="floor")
                q_pj = q_idx[pix_m] % H_p
                q_center_u = (q_pj * ps + (ps // 2)).to(torch.float32)
                q_center_v = (q_pi * ps + (ps // 2)).to(torch.float32)

                flow_u = pred_flow[pix_m, pix_i, pix_j, 0].to(torch.float32) * ps
                flow_v = pred_flow[pix_m, pix_i, pix_j, 1].to(torch.float32) * ps
                q_u = q_center_u + flow_u
                q_v = q_center_v + flow_v
                pix_q_uv = torch.stack([q_u, q_v], dim=-1)  # [K,2]
        else:
            pix_m = q_tokens_aa.new_empty((0,), dtype=torch.long)
            pix_w = q_tokens_aa.new_empty((0,), dtype=torch.float32)
            pix_t_uv = q_tokens_aa.new_empty((0, 2), dtype=torch.float32)
            pix_q_uv = q_tokens_aa.new_empty((0, 2), dtype=torch.float32)

        # -----------------------------
        # Now group into List[Dict[template_idx -> {...}]] per batch item.
        # -----------------------------
        results: List[Dict[int, Dict[str, Any]]] = []

        # Build fast access: for each m, which (b,s)
        # (small loops are fine: top_k_templates is small; M also reduced after pruning)
        for b in range(B):
            entry: Dict[int, Dict[str, Any]] = {}

            # chosen templates for this batch item (in ranked order)
            tmpl_ids = chosen_template_idx[b].tolist()
            tmpl_scores = chosen_template_score[b].tolist()

            # Precompute mask of coarse rows belonging to this b
            m_mask_b = (b_idx == b) if M > 0 else None

            # Precompute mask of refined pixels belonging to this b via pix_m -> b_idx[pix_m]
            if pix_m.numel() > 0:
                pix_b = b_idx[pix_m]
            else:
                pix_b = None

            for rank, (s, s_score) in enumerate(zip(tmpl_ids, tmpl_scores)):
                s_int = int(s)
                s_score_f = float(s_score)

                # ---- coarse rows for (b,s)
                if M > 0:
                    m_mask = m_mask_b & (s_idx == s_int)
                    m_sel = m_mask.nonzero(as_tuple=True)[0]  # [Nc]
                else:
                    m_sel = torch.empty((0,), device=self.device, dtype=torch.long)

                coarse_dict = {
                    "q_center_uv": coarse_q_center_uv[m_sel].detach(),  # [Nc,2]
                    "t_center_uv": coarse_t_center_uv[m_sel].detach(),  # [Nc,2]
                    "q_idx": q_idx[m_sel].detach(),
                    "t_idx": t_idx[m_sel].detach(),
                    "patch_score": patch_score_sel[m_sel].detach(),
                }

                # ---- refined pixels for (b,s)
                # We output per pixel and a pointer "pair_row" which is the index in the coarse arrays for that template.
                if pix_m.numel() > 0:
                    # select pixels whose coarse-pair m belongs to this (b,s)
                    # coarse m index for each pixel is pix_m[k]
                    mask_pix = (pix_b == b) & (s_idx[pix_m] == s_int)
                    k_sel = mask_pix.nonzero(as_tuple=True)[0]  # [Nr]

                    # map global m index -> local row within this template's coarse list
                    # Build mapping by: take m_sel (coarse rows), create a dict-like map via scatter
                    if m_sel.numel() > 0 and k_sel.numel() > 0:
                        # local row id for each global m in m_sel
                        local_row = torch.full((M,), -1, device=self.device, dtype=torch.long)
                        local_row[m_sel] = torch.arange(m_sel.numel(), device=self.device, dtype=torch.long)
                        pair_row = local_row[pix_m[k_sel]]
                    else:
                        pair_row = torch.empty((0,), device=self.device, dtype=torch.long)

                    refined_dict = {
                        "t_uv": pix_t_uv[k_sel].detach(),   # [Nr,2]
                        "q_uv": pix_q_uv[k_sel].detach(),   # [Nr,2]
                        "w": pix_w[k_sel].detach(),         # [Nr]
                        "pair_row": pair_row.detach(),      # [Nr] index into this template's coarse rows
                    }
                else:
                    refined_dict = {
                        "t_uv": torch.empty((0, 2), device=self.device, dtype=torch.float32),
                        "q_uv": torch.empty((0, 2), device=self.device, dtype=torch.float32),
                        "w": torch.empty((0,), device=self.device, dtype=torch.float32),
                        "pair_row": torch.empty((0,), device=self.device, dtype=torch.long),
                    }

                entry[s_int] = {
                    "template_idx": s_int,
                    "template_rank": int(rank),
                    "template_score": s_score_f,
                    "coarse": coarse_dict,
                    "refined": refined_dict,
                }

            results.append(entry)

        return results