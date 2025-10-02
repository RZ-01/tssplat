"""
Warp-compatible dataloader for TetSplatting.
Directly copied from original data loading logic.
"""

import torch
import numpy as np
import cv2
from PIL import Image
import glob
import os
from typing import Optional, Union, List, Dict
from dataclasses import dataclass

from utils.config import parse_structured, get_device
from utils.typing import *


class WarpWonder3DImgDataset:
    """Wonder3D dataset loader compatible with Warp training"""
    
    @dataclass
    class Config:
        camera_mvp_root: str
        camera_views: List[str]
        image_root: str

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        
        # Load image data (same as original)
        self.all_tgt_imgs, self.all_mvp_mats, self.all_mv_mats, self.all_campos, self.all_tgt_ns, self.all_tgt_ds = self.load_img_data()

        # Background as all white (same as original)
        self.bgs = [np.ones((self.all_tgt_imgs[0].shape[0], self.all_tgt_imgs[0].shape[1], 3))
                    for i in range(len(self.all_tgt_imgs))]

        # Camera params (same as original)
        self.camera_p = self.all_mvp_mats[0] @ np.linalg.inv(self.all_mv_mats[0])
        self.camera_dist = np.linalg.norm(self.all_campos[0])

        # Resolution and spp (same as original)
        self.resolution = self.all_tgt_imgs[0].shape[0]  # square imgs
        self.spp = 1

    def load_img_data(self):
        """Load image data - directly copied from original"""
        all_tgt_imgs = []
        all_mvp_mats = []
        all_mv_mats = []
        all_campos = []
        all_tgt_ns = []
        all_tgt_ds = []

        for view in self.cfg.camera_views:
            camera_filename = f"{self.cfg.camera_mvp_root}/{view}_mvp.npy"
            all_mvp_mats.append(np.load(camera_filename))
            all_tgt_imgs.append(np.zeros(1))
            all_tgt_ns.append(np.zeros(1))

        # Masked images
        img_root = os.path.dirname(self.cfg.image_root) + "/masked_colors1"
        for img_file in os.listdir(img_root):
            found = False
            for i, c in enumerate(self.cfg.camera_views):
                if c in img_file:
                    found = True
                    tgt_img = np.array(Image.open(os.path.join(
                        img_root, img_file))).astype(np.float32) / 255.0

                    tgt_img = cv2.resize(tgt_img, (512, 512), cv2.INTER_CUBIC)
                    tgt_img[..., 3] = np.where(tgt_img[..., 3] < 0.8, 0, 1)

                    all_tgt_imgs[i] = tgt_img
                    break

            assert found

        # Normal images
        img_root = os.path.dirname(self.cfg.image_root) + "/normals"
        for img_file in os.listdir(img_root):
            found = False
            for i, c in enumerate(self.cfg.camera_views):
                if c in img_file:
                    found = True
                    tgt_img = np.array(Image.open(os.path.join(
                        img_root, img_file))).astype(np.float32) / 255.0

                    tgt_img = cv2.resize(tgt_img, (512, 512), cv2.INTER_CUBIC)
                    tgt_img[..., 0:3] = (tgt_img[..., 0:3] - 0.5) * 2

                    all_tgt_ns[i] = tgt_img
                    break

            assert found

        # Process loaded data
        imgs = []
        mvps = []
        ns = []
        ds = []
        for i, (mvp, img, n) in enumerate(zip(all_mvp_mats, all_tgt_imgs, all_tgt_ns)):
            if len(img.shape) == 3:
                imgs.append(img)
                ds.append(img[..., -1:])
                ns.append(n)
                mvps.append(mvp.astype(np.float32))
                all_campos.append(np.asarray([0, 0, 1]))

        all_tgt_imgs = imgs
        all_tgt_ns = ns
        all_tgt_ds = ds
        all_mvp_mats = mvps
        all_mv_mats = all_mvp_mats
        
        return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds


class WarpMistubaImgDataset:
    """Mitsuba dataset loader compatible with Warp training"""
    
    @dataclass
    class Config:
        image_root: str

    def __init__(self, cfg):  # Accept direct config object, not parsed dict
        # cfg should be already parsed MistubaImgDataset.Config
        self.cfg = cfg
        
        # Load image data (same as original)
        self.all_tgt_imgs, self.all_mvp_mats, self.all_mv_mats, self.all_campos, self.all_tgt_ns, self.all_tgt_ds = self.load_img_data()

        # Background as all white (same as original)
        self.bgs = [np.ones((self.all_tgt_imgs[0].shape[0], self.all_tgt_imgs[0].shape[1], 3))
                    for i in range(len(self.all_tgt_imgs))]

        # Camera params (same as original)
        self.camera_p = self.all_mvp_mats[0] @ np.linalg.inv(self.all_mv_mats[0])
        self.camera_dist = np.linalg.norm(self.all_campos[0])

        # Resolution and spp (same as original)
        self.resolution = self.all_tgt_imgs[0].shape[0]  # square imgs
        self.spp = 1

    def load_img_data(self):
        """Load image data - directly copied from original"""
        all_tgt_imgs = []
        all_mvp_mats = []
        all_mv_mats = []
        all_campos = []
        all_tgt_ns = []
        all_tgt_ds = []

        assert os.path.isdir(self.cfg.image_root)
        tgt_path = self.cfg.image_root
        
        for img_file in glob.glob(os.path.join(tgt_path, "img*rgba*.png")):
            tgt_img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
            all_tgt_imgs.append(tgt_img)

            img_id = os.path.basename(img_file).split(".")[0].split("_")[-1]

            mvp_mat_file = os.path.join(tgt_path, "mvp_mtx_{}.npy".format(img_id))
            mvp_mat = np.load(mvp_mat_file)
            all_mvp_mats.append(mvp_mat)

            mv_file = os.path.join(tgt_path, "mv_{}.npy".format(img_id))
            mv = np.load(mv_file)
            all_mv_mats.append(mv)

            campos = np.linalg.inv(mv)[:3, 3]
            all_campos.append(campos)

            normal_file = os.path.join(tgt_path, "normal_{}.npy".format(img_id))
            if os.path.exists(normal_file):
                n = np.load(normal_file)
            else:
                n = np.zeros_like(tgt_img)
            all_tgt_ns.append(n)

            depth_file = os.path.join(tgt_path, "depth_{}.npy".format(img_id))
            if os.path.exists(depth_file):
                d = np.load(depth_file)
                d = d[..., None]
            else:
                d = np.zeros_like(tgt_img)
            all_tgt_ds.append(d)

            # Validation checks (same as original)
            try:
                assert np.all(np.isfinite(tgt_img))
                assert np.all(np.isfinite(mvp_mat))
                assert np.all(np.isfinite(mv))
                assert np.all(np.isfinite(campos))
                assert np.all(np.isfinite(d))
            except:
                import pdb
                pdb.set_trace()

        return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds


class WarpBlenderImgDataset:
    """Blender dataset loader for Warp training (simplified version)"""
    
    @dataclass
    class Config:
        image_root: str
        resolution: int = 512

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        
        # Load image data 
        self.all_tgt_imgs, self.all_mvp_mats, self.all_mv_mats, self.all_campos, self.all_tgt_ns, self.all_tgt_ds = self.load_img_data()

        # Background as all white
        self.bgs = [np.ones((self.cfg.resolution, self.cfg.resolution, 3))
                    for i in range(len(self.all_tgt_imgs))]

        # Camera params
        if len(self.all_mvp_mats) > 0:
            self.camera_p = self.all_mvp_mats[0] @ np.linalg.inv(self.all_mv_mats[0])
            self.camera_dist = np.linalg.norm(self.all_campos[0])
        else:
            # Default values
            self.camera_p = np.eye(4)
            self.camera_dist = 1.0

        # Resolution and spp
        self.resolution = self.cfg.resolution
        self.spp = 1

    def load_img_data(self):
        """Load Blender-style data"""
        all_tgt_imgs = []
        all_mvp_mats = []
        all_mv_mats = []
        all_campos = []
        all_tgt_ns = []
        all_tgt_ds = []

        if not os.path.isdir(self.cfg.image_root):
            print(f"Warning: Image root {self.cfg.image_root} not found")
            return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds

        # Look for transforms.json or similar metadata
        import json
        transforms_path = os.path.join(self.cfg.image_root, "transforms.json")
        if os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                meta = json.load(f)
            
            for frame in meta.get('frames', []):
                # Load image
                img_path = os.path.join(self.cfg.image_root, frame['file_path'])
                if img_path.endswith('.png'):
                    img_path = img_path
                else:
                    img_path = img_path + '.png'
                
                if os.path.exists(img_path):
                    img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
                    img = cv2.resize(img, (self.cfg.resolution, self.cfg.resolution), cv2.INTER_CUBIC)
                    all_tgt_imgs.append(img)
                    
                    # Transform matrix
                    transform_matrix = np.array(frame['transform_matrix'])
                    all_mv_mats.append(transform_matrix)
                    
                    # For MVP, we'd need projection matrix (using identity for now)
                    proj_matrix = np.eye(4)  # Simplified
                    mvp_matrix = proj_matrix @ transform_matrix
                    all_mvp_mats.append(mvp_matrix.astype(np.float32))
                    
                    # Camera position
                    campos = transform_matrix[:3, 3]
                    all_campos.append(campos)
                    
                    # Dummy normal and depth
                    all_tgt_ns.append(np.zeros_like(img))
                    all_tgt_ds.append(img[..., -1:] if img.shape[-1] > 3 else np.ones_like(img[..., :1]))

        return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds


class WarpDataLoader:
    """Warp-compatible dataloader - directly copied logic from original"""
    
    @dataclass
    class Config:
        batch_size: int
        total_num_iter: int
        world_size: int = 1
        rank: int = 0

    def __init__(self, dataset, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.dataset = dataset
        
        # Convert to torch tensors (same as original)
        self.to_torch()
        
        # Set batch properties
        self.n_images = len(self.dataset.all_tgt_imgs)
        self.num_forward_per_iter = max(1, self.cfg.batch_size // self.n_images)

    def to_torch(self):
        """Convert numpy arrays to torch tensors - directly copied from original"""
        self.img = torch.tensor(np.array(self.dataset.all_tgt_imgs),
                               dtype=torch.float32).to(self.device)
        self.n = torch.tensor(np.array(self.dataset.all_tgt_ns),
                             dtype=torch.float32).to(self.device)
        self.d = torch.tensor(np.array(self.dataset.all_tgt_ds),
                             dtype=torch.float32).to(self.device)
        self.mv = torch.tensor(np.array(self.dataset.all_mv_mats),
                              dtype=torch.float32).to(self.device)
        self.campos = torch.tensor(np.array(self.dataset.all_campos),
                                  dtype=torch.float32).to(self.device)
        self.mvp = torch.tensor(np.array(self.dataset.all_mvp_mats),
                               dtype=torch.float32).to(self.device)
        self.bg = torch.tensor(np.array(self.dataset.bgs),
                              dtype=torch.float32).to(self.device)

        # Combine background with images (same as original)
        self.img = torch.cat(
            (torch.lerp(self.bg, self.img[..., 0:3], self.img[..., 3:4]), self.img[..., 3:4]), 
            dim=-1
        )

        self.iter_res = self.dataset.resolution
        self.iter_spp = self.dataset.spp

    def __call__(self, iter_num: int, forward_id: int = 0) -> Dict[str, torch.Tensor]:
        """Get batch data - same logic as original"""
        # Simple round-robin selection for now
        idx = (iter_num * self.num_forward_per_iter + forward_id) % self.n_images
        
        return {
            "img": self.img[idx:idx+1],  # Add batch dimension
            "n": self.n[idx:idx+1],
            "d": self.d[idx:idx+1], 
            "mv": self.mv[idx:idx+1],
            "campos": self.campos[idx:idx+1],
            "mvp": self.mvp[idx:idx+1],
            "background": self.bg[idx:idx+1],
            "resolution": self.iter_res
        }

    def __len__(self):
        return self.n_images


class WarpMistubaImgDataLoader:
    """Warp version of MistubaImgDataLoader - matches original structure"""
    
    @dataclass 
    class Config:
        batch_size: int
        total_num_iter: int
        world_size: int = 1
        rank: int = 0
        dataset_config: WarpMistubaImgDataset.Config = None

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        
        # Create dataset using dataset_config
        self.dataset = WarpMistubaImgDataset(self.cfg.dataset_config)
        
        # Prepare data (same as original)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data - copied from original"""
        # Convert to torch tensors (same as original DataLoader.to_torch)
        self.img = torch.tensor(np.array(self.dataset.all_tgt_imgs),
                               dtype=torch.float32).to(self.device)
        self.n = torch.tensor(np.array(self.dataset.all_tgt_ns),
                             dtype=torch.float32).to(self.device)
        self.d = torch.tensor(np.array(self.dataset.all_tgt_ds),
                             dtype=torch.float32).to(self.device)
        self.mv = torch.tensor(np.array(self.dataset.all_mv_mats),
                              dtype=torch.float32).to(self.device)
        self.campos = torch.tensor(np.array(self.dataset.all_campos),
                                  dtype=torch.float32).to(self.device)
        self.mvp = torch.tensor(np.array(self.dataset.all_mvp_mats),
                               dtype=torch.float32).to(self.device)
        self.bg = torch.tensor(np.array(self.dataset.bgs),
                              dtype=torch.float32).to(self.device)
        
        # Apply background blending (same as original)
        self.img = torch.cat(
            (torch.lerp(self.bg, self.img[..., 0:3], self.img[..., 3:4]), self.img[..., 3:4]), dim=-1)
        
        # Set properties
        self.n_images = len(self.dataset.all_tgt_imgs)
        self.num_forward_per_iter = max(1, self.cfg.batch_size // self.n_images)
        self.resolution = self.dataset.resolution
        
    def __len__(self):
        return self.n_images
    
    def __call__(self, iter_num, forw_id):
        """Generate batch - same logic as original"""
        with torch.no_grad():
            # Random indices selection (same as original)
            indices = np.random.randint(0, self.n_images, size=self.cfg.batch_size)
            
            batch = {
                "img": self.img[indices],
                "n": self.n[indices] if hasattr(self, 'n') else None,
                "d": self.d[indices] if hasattr(self, 'd') else None,
                "mvp": self.mvp[indices],
                "mv": self.mv[indices],
                "campos": self.campos[indices],
                "background": self.bg[indices],
                "resolution": self.resolution
            }
        
        return batch


def load_warp_dataloader(dataset_type: str):
    """Factory function to load appropriate dataset type"""
    if dataset_type == "wonder3d":
        return WarpWonder3DImgDataset
    elif dataset_type == "mitsuba":
        # For now, use Wonder3D dataset as fallback since MistubaImgDataset is not implemented
        print("Warning: Using WarpWonder3DImgDataset as fallback for MistubaImgDataLoader")
        return WarpWonder3DImgDataset
    elif dataset_type == "blender":
        return WarpBlenderImgDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                        f"Supported types: wonder3d, mitsuba, blender")