from utils import eval_errors
from utils import misc

def calc_mssd(trans_m2oc, trans_m2oc_gt, camera_c2w, object_mesh_vertices, object_syms, object_diameter=1.0):
    R_est, t_est = trans_m2oc[:3, :3], trans_m2oc[:3, 3:]
    R_gt, t_gt = trans_m2oc_gt[:3, :3], trans_m2oc_gt[:3, 3:]
    K = misc.get_intrinsic_matrix(camera_c2w)

    # Normalize MSSD by object diameter.
    mssd_e, mssd_id = eval_errors.mssd(
        R_est, t_est, R_gt, t_gt, object_mesh_vertices, object_syms
    )

    mssd = mssd_e / object_diameter

    return mssd

    
def calc_mspd(trans_m2oc, trans_m2oc_gt, camera_c2w, object_mesh_vertices, object_syms, object_diameter=1.0):
    R_est, t_est = trans_m2oc[:3, :3], trans_m2oc[:3, 3:]
    R_gt, t_gt = trans_m2oc_gt[:3, :3], trans_m2oc_gt[:3, 3:]
    K = misc.get_intrinsic_matrix(camera_c2w)
    # Normalize MSPD by image width.
    mspd_e, mspd_id = eval_errors.mspd(
        R_est, t_est, R_gt, t_gt, K, object_mesh_vertices, object_syms
    )
    mspd = mspd_e / object_diameter
    return mspd


def maximum_symmetry_aware_errors(
        trans_m2oc, trans_m2oc_gt, camera_c2w, object_mesh_vertices, object_syms, object_diameter=1.0
):
    mssd = calc_mssd(trans_m2oc, trans_m2oc_gt, camera_c2w, object_mesh_vertices, object_syms, object_diameter)
    mspd = calc_mspd(trans_m2oc, trans_m2oc_gt, camera_c2w, object_mesh_vertices, object_syms, object_diameter)
    return mssd, mspd