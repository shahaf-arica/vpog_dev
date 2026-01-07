import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import get_logger
from huggingface_hub import snapshot_download

logger = get_logger(__name__)

@hydra.main(
    version_base=None,
    config_path="../../training/config/user",
    config_name="hf",
)
def download(cfg: DictConfig) -> None:
    target_path = Path(cfg.saving_dir)
    hf_token  = cfg.token

    target_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="bop-benchmark/megapose",
        repo_type="dataset",
        revision="main",
        allow_patterns=[
            # "MegaPose-GSO/*",          # first subfolder
            # "MegaPose-ShapeNetCore/*",  # second subfolder
            "MegaPose-GSO-fixed/*",          # first subfolder fix
            "MegaPose-ShapeNetCore-fixed/*"  # second subfolder fix
        ],
        local_dir=target_path,  # where to save them
        local_dir_use_symlinks=False,   # actually copy the files
        resume_download=True,
        token=hf_token
    )


if __name__ == "__main__":
    download()
