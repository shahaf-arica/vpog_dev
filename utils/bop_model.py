
import bop_toolkit_lib.config as bop_config
from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.misc as bop_misc
from utils import structs
from utils import misc as misc_util
import numpy as np
import open3d as o3d


class BOPModels:
    def __init__(self, dataset_name, datasets_path=None):
        datasets_path = bop_config.datasets_path if datasets_path is None else datasets_path
        self.dataset_name = dataset_name
        bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=dataset_name)
        self.object_lids = bop_model_props["obj_ids"]
        self.models_info = inout.load_json(bop_model_props["models_info_path"], keys_to_int=True)
        self.model_tpath = bop_model_props["model_tpath"]
        # create general renderer object
        self.object_syms = {}
        for object_lid in self.object_lids:
            self.object_syms[object_lid] = bop_misc.get_symmetry_transformations(
                self.models_info[object_lid], max_sym_disc_step= 0.01
            )

        self.model_diameters = {}
        for object_lid in self.object_lids:
            self.model_diameters[object_lid] = self.model_diameter(object_lid)

    def model_path(self, object_lid):
        return self.model_tpath.format(obj_id=object_lid)
    
    def pcd(self, object_lid):
        pcd_model = o3d.io.read_point_cloud(self.model_path(object_lid))
        return pcd_model
    
    def mesh(self, object_lid):        
        mesh_model = o3d.io.read_triangle_mesh(self.model_path(object_lid))
        return mesh_model
    
    def vertices(self, object_lid):
        pcd_model = self.pcd(object_lid)
        return np.asarray(pcd_model.points)
    
    def transformed_pcd(self, object_lid, R_m2c, t_m2c):
        """
        Note: camera_c2w is used only for intrinsics the extrensic projection determined by R_m2c and t_m2c
        """
        pcd_model = self.pcd(object_lid)
        # transform the model to camera coordinate
        pcd_model.transform(misc_util.get_rigid_matrix(structs.RigidTransform(R=R_m2c,t=t_m2c)))
        return pcd_model
    
    def model_diameter(self, object_lid):
        pcd_model = self.pcd(object_lid)
        # find scale of the model
        object_diameter =  np.linalg.norm(pcd_model.get_axis_aligned_bounding_box().get_extent()) 
        return object_diameter
    
    def get_object_syms(self, object_lid):
        return bop_misc.get_symmetry_transformations(
                self.models_info[object_lid], max_sym_disc_step= 0.01
        )