# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import warnings
from os import path
from pathlib import Path
from typing import Dict, List, Optional
import random

import numpy as np
import torch
from PIL import Image
from pytorch3d.common.datatypes import Device
from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.renderer import HardPhongShader
from tabulate import tabulate
from pytorch3d.datasets.r2n2 import utils
from pytorch3d.datasets.r2n2.utils import (
    BlenderCamera,
    align_bbox,
    compute_extrinsic_matrix,
    read_binvox_coords,
    voxelize,
)
import utils_vox


SYNSET_DICT_DIR = Path(utils.__file__).resolve().parent
MAX_CAMERA_DISTANCE = 1.75  # Constant from R2N2.
VOXEL_SIZE = 128
# Intrinsic matrix extracted from Blender. Taken from meshrcnn codebase:
# https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
BLENDER_INTRINSIC = torch.tensor(
    [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
)


class R2N2(ShapeNetBase):  # pragma: no cover
    """
    This class loads the R2N2 dataset from a given directory into a Dataset object.
    The R2N2 dataset contains 13 categories that are a subset of the ShapeNetCore v.1
    dataset. The R2N2 dataset also contains its own 24 renderings of each object and
    voxelized models.

    This is an OPTIMIZED version that only loads a single random view per model
    instead of loading all views and then subsampling.
    """

    def __init__(
        self,
        split: str,
        shapenet_dir,
        r2n2_dir,
        splits_file,
        return_voxels: bool = False,
        return_feats: bool = False,
        views_rel_path: str = "ShapeNetRendering",
        voxels_rel_path: str = "ShapeNetVoxels",
        load_textures: bool = False,
        texture_resolution: int = 4,
    ) -> None:
        """
        Store each object's synset id and models id the given directories.

        Args:
            split (str): One of (train, val, test).
            shapenet_dir (path): Path to ShapeNet core v1.
            r2n2_dir (path): Path to the R2N2 dataset.
            splits_file (path): File containing the train/val/test splits.
            return_voxels(bool): Indicator of whether or not to return voxels as a tensor
                of shape (D, D, D) where D is the number of voxels along each dimension.
            return_feats(bool): Indicator of whether image features from a pretrained resnet18
                are also returned in the dataloader or not.
            views_rel_path: path to rendered views within the r2n2_dir. If not specified,
                the renderings are assumed to be at os.path.join(rn2n_dir, "ShapeNetRendering").
            voxels_rel_path: path to rendered views within the r2n2_dir. If not specified,
                the renderings are assumed to be at os.path.join(rn2n_dir, "ShapeNetVoxels").
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = shapenet_dir
        self.r2n2_dir = r2n2_dir
        self.views_rel_path = views_rel_path
        self.voxels_rel_path = voxels_rel_path
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.return_feats = return_feats

        if split not in ["train", "val", "test"]:
            raise ValueError("split has to be one of (train, val, test).")

        with open(
            path.join(SYNSET_DICT_DIR, "r2n2_synset_dict.json"), "r"
        ) as read_dict:
            self.synset_dict = json.load(read_dict)
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        with open(splits_file) as splits:
            split_dict = json.load(splits)[split]

        self.return_images = True
        if not path.isdir(path.join(r2n2_dir, views_rel_path)):
            self.return_images = False
            warnings.warn(
                f"{views_rel_path} not found in {r2n2_dir}. "
                "R2N2 renderings will be skipped."
            )

        self.return_voxels = return_voxels
        if not path.isdir(path.join(r2n2_dir, voxels_rel_path)):
            self.return_voxels = False
            warnings.warn(
                f"{voxels_rel_path} not found in {r2n2_dir}. "
                "Voxel coordinates will be skipped."
            )

        synset_set = set()
        self.views_per_model_list = []
        synset_num_instances = []

        for synset in split_dict.keys():
            if not (
                path.isdir(path.join(shapenet_dir, synset))
                and synset in self.synset_dict
            ):
                warnings.warn(
                    f"Synset category {synset} from splits file is not "
                    f"in {shapenet_dir} or not part of R2N2 dataset."
                )
                continue

            synset_set.add(synset)
            self.synset_start_idxs[synset] = len(self.synset_ids)
            model_count_for_synset = 0
            
            for model in split_dict[synset]:
                shapenet_path = path.join(shapenet_dir, synset, model)
                if not path.isdir(shapenet_path):
                    warnings.warn(
                        f"Model {model} from category {synset} is not present "
                        f"in {shapenet_dir}. Skipping."
                    )
                    continue
                
                self.synset_ids.append(synset)
                self.model_ids.append(model)
                self.views_per_model_list.append(split_dict[synset][model])
                model_count_for_synset += 1

            synset_num_instances.append(
                (self.synset_dict[synset], model_count_for_synset)
            )
            self.synset_num_models[synset] = model_count_for_synset

        headers = ["category", "#models"]
        synset_num_instances.append(("total", sum(n for _, n in synset_num_instances)))
        print(
            "Found the following number of models for each category:"
        )
        print(
            tabulate(synset_num_instances, headers, numalign="left", stralign="center")
        )

        synset_not_present = [
            self.synset_inv.pop(self.synset_dict[synset])
            for synset in self.synset_dict
            if synset not in synset_set
        ]
        if len(synset_not_present) > 0:
            warnings.warn(
                f"The following categories are in R2N2's official mapping but not "
                f"found in the dataset location {shapenet_dir}: {', '.join(synset_not_present)}"
            )

    def __getitem__(self, idx) -> Dict:
        """
        Read a model by the given index. This method is optimized to load only one
        random view and its corresponding data, avoiding the overhead of loading all views.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3).
            - synset_id (str): synset id.
            - model_id (str): model id.
            - label (str): synset label.
            - images: FloatTensor of shape (H, W, C) of a single rendering.
            - R: Rotation matrix of shape (3, 3).
            - T: Translation matrix of shape (3).
            - K: Intrinsic matrix of shape (4, 4).
            - voxels: Voxels of shape (D, D, D).
            - feats: (Optional) FloatTensor of shape (F,).
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], "model.obj"
        )
        try:
            verts, faces, _ = self._load_mesh(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Mesh file not found at {model_path}")
        
        model["verts"] = verts
        model["faces"] = faces
        model["label"] = self.synset_dict[model["synset_id"]]

        # OPTIMIZATION: Decide which view to load FIRST.
        available_views = self.views_per_model_list[idx]
        view_to_load = random.choice(available_views)

        if self.return_images:
            rendering_path = path.join(
                self.r2n2_dir, self.views_rel_path,
                model["synset_id"], model["model_id"], "rendering"
            )

            # OPTIMIZATION: Load only the single image needed.
            image_path = path.join(rendering_path, f"{view_to_load:02d}.png")
            try:
                raw_img = Image.open(image_path)
                image = torch.from_numpy(np.array(raw_img) / 255.0)[..., :3]
                model["images"] = image.to(dtype=torch.float32)
            except FileNotFoundError:
                 raise FileNotFoundError(f"Image file not found at {image_path}")

            # OPTIMIZATION: Load features and select only the single feature needed.
            if self.return_feats:
                feats_path = path.join(rendering_path, "feats.npy")
                try:
                    all_feats = torch.from_numpy(np.load(feats_path))
                    model["feats"] = all_feats[view_to_load].to(dtype=torch.float32)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Features file not found at {feats_path}")

            # OPTIMIZATION: Read metadata and calculate camera for the single view.
            metadata_path = path.join(rendering_path, "rendering_metadata.txt")
            try:
                with open(metadata_path, "r") as f:
                    metadata_lines = f.readlines()
                
                azim, elev, _, dist_ratio, _ = [
                    float(v) for v in metadata_lines[view_to_load].strip().split(" ")
                ]
                dist = dist_ratio * MAX_CAMERA_DISTANCE
                RT = compute_extrinsic_matrix(azim, elev, dist)
                R, T = self._compute_camera_calibration(RT)
                
                model["R"] = R
                model["T"] = T

            except (FileNotFoundError, IndexError):
                 raise ValueError(f"Metadata file not found or incomplete at {metadata_path}")
            
            # Intrinsic matrix is constant for all views.
            model["K"] = torch.tensor([
                [2.1875, 0.0, 0.0, 0.0],
                [0.0, 2.1875, 0.0, 0.0],
                [0.0, 0.0, -1.002002, -0.2002002],
                [0.0, 0.0, 1.0, 0.0],
            ])

        if self.return_voxels:
            voxel_path = path.join(
                self.r2n2_dir, self.voxels_rel_path,
                model["synset_id"], model["model_id"], "model.binvox"
            )
            if not path.isfile(voxel_path):
                raise FileNotFoundError(
                    f"Voxel file not found for model {model['model_id']} "
                    f"from category {model['synset_id']}."
                )

            with open(voxel_path, "rb") as f:
                voxel_coords = read_binvox_coords(f)
            
            voxel_coords = align_bbox(voxel_coords, model["verts"])
            model["voxel_coords"] = voxel_coords
            model["voxels"] = utils_vox.voxelize_xyz(
                voxel_coords.unsqueeze(0), 32, 32, 32
            ).squeeze(0)

        return model

    def _compute_camera_calibration(self, RT):
        """
        Helper function for calculating rotation and translation matrices from ShapeNet
        to camera transformation and ShapeNet to PyTorch3D transformation.
        """
        shapenet_to_pytorch3d = torch.tensor([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        
        RT_pytorch3d = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)
        R = RT_pytorch3d[:3, :3]
        T = RT_pytorch3d[3, :3]
        return R, T
    
    # NOTE: The render method is kept for utility, but it is not optimized for
    # the single-view loading strategy of __getitem__. It may be slow if used
    # to render many views.
    def render(
        self,
        model_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sample_nums: Optional[List[int]] = None,
        idxs: Optional[List[int]] = None,
        view_idxs: Optional[List[int]] = None,
        shader_type=HardPhongShader,
        device: Device = "cpu",
        **kwargs
    ) -> torch.Tensor:
        """
        Render models with BlenderCamera by default to achieve the same orientations as the
        R2N2 renderings.
        """
        if view_idxs is None:
            # Default to rendering the first view if not specified
            view_idxs = [0] 

        idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        
        # This part is inefficient as it calls the slow parts of the original logic
        # It's kept for API compatibility.
        all_r, all_t, all_k = [], [], []
        for i in idxs:
            model = self._get_item_ids(i)
            rendering_path = path.join(
                self.r2n2_dir, self.views_rel_path, model["synset_id"], model["model_id"], "rendering"
            )
            metadata_path = path.join(rendering_path, "rendering_metadata.txt")
            with open(metadata_path, "r") as f:
                metadata_lines = f.readlines()
            
            for v_idx in view_idxs:
                azim, elev, _, dist_ratio, _ = [
                    float(v) for v in metadata_lines[v_idx].strip().split(" ")
                ]
                dist = dist_ratio * MAX_CAMERA_DISTANCE
                RT = compute_extrinsic_matrix(azim, elev, dist)
                R, T = self._compute_camera_calibration(RT)
                all_r.append(R)
                all_t.append(T)
                all_k.append(torch.tensor([
                    [2.1875, 0.0, 0.0, 0.0], [0.0, 2.1875, 0.0, 0.0],
                    [0.0, 0.0, -1.002002, -0.2002002], [0.0, 0.0, 1.0, 0.0],
                ]))

        r = torch.stack(all_r)
        t = torch.stack(all_t)
        k = torch.stack(all_k)

        blend_cameras = BlenderCamera(
            R=kwargs.get("R", r), T=kwargs.get("T", t),
            K=kwargs.get("K", k), device=device,
        )
        cameras = kwargs.get("cameras", blend_cameras).to(device)
        kwargs.pop("cameras", None)
        
        return super().render(
            idxs=idxs, shader_type=shader_type, device=device, cameras=cameras, **kwargs
        )

