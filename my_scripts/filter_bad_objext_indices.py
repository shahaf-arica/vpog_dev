import os
import json
import glob
from tqdm import tqdm
if __name__ == "__main__":
    template_root_dir = "datasets/templates/shapenet"

    templates_dirs = glob.glob(f"{template_root_dir}/*")
    templates_dirs.sort()
    templates_dirs = [d for d in templates_dirs if not "object" in d]
    bad_dirs = []
    for d in tqdm(templates_dirs, desc="Checking existing template dirs"):
        if len(glob.glob(f"{d}/*")) < 324:
            bad_dirs.append(d)

    indices_path = "datasets/shapenet/train_pbr_web/object_index.json"

    # Load existing indices
    with open(indices_path, 'r') as f:
        obj_indices = json.load(f)

    new_obj_indices = []

    