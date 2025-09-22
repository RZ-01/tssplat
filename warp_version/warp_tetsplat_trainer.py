"""
Complete Warp-based TetSplatting trainer.
Directly replicates the original trainer.py logic but uses Warp components.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import trange, tqdm
from typing import Optional, Dict, Any
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from geometry import load_geometry
from materials import load_material
from utils.config import load_config
from utils.optimizer import AdamUniform

from renderers import MeshRasterizer

from warp_dataloader import WarpDataLoader, load_warp_dataloader
from warp_image_loss import WarpImageLoss
from warp_tetmesh import WarpTetMesh, load_tetmesh_from_surface_mesh


class LinearInterpolateScheduler:
    """Linear interpolation scheduler - directly copied from original"""
    
    def __init__(self, start_iter, end_iter, start_val, end_val, freq):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_val = start_val
        self.end_val = end_val
        self.freq = freq

    def __call__(self, iter):
        if iter < self.start_iter or iter % self.freq != 0 or iter == 0:
            return None

        p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_val * (1 - p) + self.end_val * p


class WarpTetMeshGeometry:
    """Warp-based geometry wrapper for compatibility with original trainer and renderer"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.optimize_geo = getattr(cfg, 'optimize_geo', True)
        self.output_path = getattr(cfg, 'output_path', 'results')
        
        # Load initial tetrahedral mesh
        init_mesh_path = getattr(cfg, 'init_spheres_path', '../mesh_data/s.1.obj')
        
        if os.path.exists(init_mesh_path):
            self.tetmesh = load_tetmesh_from_surface_mesh(init_mesh_path)
        else:
            # Create default sphere
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            self.tetmesh = load_tetmesh_from_surface_mesh(sphere)
        
        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tet_v = torch.from_numpy(self.tetmesh.vertices).float().to(device)
        self.tet_v.requires_grad_(True)
        
        self.tet_elements = torch.from_numpy(self.tetmesh.elements).long().to(device)
        self.rest_matrices = torch.from_numpy(self.tetmesh.rest_matrices).float().to(device)
        self.surface_vid = torch.from_numpy(self.tetmesh.surface_vertices).long().to(device)
        self.surface_fid = torch.from_numpy(self.tetmesh.surface_faces).long().to(device)
        
        # For compatibility with original renderer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ones_surface_v = torch.ones([self.surface_vid.shape[0], 1]).to(device)
        self.zeros_surface_v = torch.zeros([self.surface_vid.shape[0], 1]).to(device)
        
        print(f"WarpTetMeshGeometry initialized: {len(self.tetmesh.vertices)} vertices, {len(self.tetmesh.elements)} tetrahedra")
    
    def __call__(self, **kwargs):
        """Forward pass to get geometry data - compatible with original renderer"""
        from warp_tetmesh import WarpBarrierEnergy
        
        # Compute barrier energy using Warp
        barrier_energy = WarpBarrierEnergy.apply(
            self.tet_v,
            self.tet_elements.flatten().int(),
            self.rest_matrices,
            1e-4,  # smoothness weight
            1e-4,  # barrier weight
            2      # barrier order
        )
        
        # Return geometry data in expected format for original renderer
        class GeometryData:
            def __init__(self, v_pos, t_pos_idx, smooth_barrier_energy):
                self.v_pos = v_pos  # Surface vertex positions
                self.t_pos_idx = t_pos_idx  # Surface face indices
                self.smooth_barrier_energy = smooth_barrier_energy
            
            def _compute_vertex_normal(self):
                """Compute vertex normals for surface vertices"""
                # Simple normal computation using face normals
                faces = self.t_pos_idx
                vertices = self.v_pos
                
                # Get face vertices
                v0 = vertices[faces[:, 0]]
                v1 = vertices[faces[:, 1]] 
                v2 = vertices[faces[:, 2]]
                
                # Compute face normals
                face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
                face_normals = torch.nn.functional.normalize(face_normals, dim=1)
                
                # Average face normals to get vertex normals
                vertex_normals = torch.zeros_like(vertices)
                for i in range(faces.shape[0]):
                    vertex_normals[faces[i, 0]] += face_normals[i]
                    vertex_normals[faces[i, 1]] += face_normals[i]
                    vertex_normals[faces[i, 2]] += face_normals[i]
                
                vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
                return vertex_normals
        
        return GeometryData(
            v_pos=self.tet_v[self.surface_vid],  # Surface vertices only
            t_pos_idx=self.surface_fid,  # Surface faces
            smooth_barrier_energy=barrier_energy
        )
    
    def parameters(self):
        """Return parameters for optimizer"""
        return [self.tet_v]
    
    def export(self, path, name, save_npy=False):
        """Export current mesh"""
        os.makedirs(path, exist_ok=True)
        
        # Update tetmesh with current vertices
        vertices_np = self.tet_v.detach().cpu().numpy()
        current_tetmesh = WarpTetMesh(
            vertices_np,
            self.tetmesh.elements,
            self.tetmesh.surface_vertices,
            self.tetmesh.surface_faces
        )
        
        # Export surface mesh
        mesh_path = os.path.join(path, f"{name}.obj")
        current_tetmesh.export_mesh(mesh_path)
        
        if save_npy:
            np.save(os.path.join(path, f"{name}_vertices.npy"), vertices_np)


def train_warp_tetsplat(cfg):
    """Main training function - directly copied logic from original trainer.py"""
    verbose = cfg.get("verbose", False)
    
    # Setup material
    material = None
    os.makedirs(os.path.join(cfg.output_path, "final/"), exist_ok=True)
    
    # Loss function setup (same as original)
    shade_loss = torch.nn.MSELoss()
    if cfg.get("fitting_stage", None) == "texture":
        assert cfg.get("material", None) is not None
        material = load_material(cfg.material_type)(cfg.material)
        shade_loss = torch.nn.L1Loss()
    
    # Load geometry
    if hasattr(cfg, 'geometry_type'):
        # Use original geometry loader
        cfg.geometry.optimize_geo = True
        cfg.geometry.output_path = cfg.output_path
        geometry = load_geometry(cfg.geometry_type)(cfg.geometry)
    else:
        # Use Warp geometry
        geometry = WarpTetMeshGeometry(cfg.geometry if hasattr(cfg, 'geometry') else cfg)
    
    # Use renderer (original or simplified)
    renderer = MeshRasterizer(geometry, material, cfg.renderer if hasattr(cfg, 'renderer') else {})
    
    # Use Warp dataloader
    dataset_class = load_warp_dataloader(cfg.dataloader_type)
    dataset = dataset_class(cfg.data)
    dataloader = WarpDataLoader(dataset, cfg.data)
    
    # Setup loss function
    loss_fn = WarpImageLoss()
    
    num_forward_per_iter = dataloader.num_forward_per_iter
    
    # Optimizer
    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.total_num_iter * num_forward_per_iter, eta_min=1e-4)
    
    # Permute surface scheduler (same as original)
    permute_surface_scheduler = None
    if cfg.get('use_permute_surface_v', False):
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    # Training state tracking
    best_loss = 1e10
    best_loss_iter = 0
    best_opt_imgs = None
    best_v = None
    
    print("Starting Warp TetSplatting training...")
    print(f"Total iterations: {cfg.total_num_iter}")
    print(f"Forward passes per iteration: {num_forward_per_iter}")
    print(f"Dataset size: {len(dataloader)}")
    
    # Main training loop - directly copied from original
    for it in trange(cfg.total_num_iter):
        for forw_id in range(num_forward_per_iter):
            batch = dataloader(it, forw_id)
            
            color_ref = batch["img"]
            
            # Depth fitting setup (same as original)
            fit_depth = cfg.get("fit_depth", False)
            if fit_depth:
                fit_depth = cfg.get("fit_depth_starting_iter", 0) < it
            
            # Renderer input (same as original)
            renderer_input = {
                "mvp": batch["mvp"],
                "only_alpha": cfg.get("fitting_stage", None) == "geometry",
                "iter_num": it,
                "resolution": batch["resolution"],
                "background": batch["background"],
                "permute_surface_scheduler": permute_surface_scheduler,
                "fit_depth": fit_depth,
                "campos": batch["campos"],
            }
            
            # Forward pass
            out = renderer(**renderer_input)
            
            # Compute losses - using original trainer logic
            img_loss = None
            if cfg.get("fitting_stage", None) == "geometry":
                img_loss = shade_loss(out["shaded"][..., -1],
                                    color_ref[0][..., -1])  # Alpha channel only
            else:
                img_loss = shade_loss(
                    out["shaded"][..., :3], color_ref[0][..., :3])  # RGB channels
            img_loss *= 20
            
            # Depth loss
            if fit_depth:
                img_loss += shade_loss(out["d"][..., -1] * color_ref[0][..., -1],
                                     batch["d"][0][..., -1] * color_ref[0][..., -1]) * 100
            
            # Regularization loss
            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out["geo_regularization"]
            
            total_loss = img_loss * 100 + reg_loss
            
            # Logging (same format as original)
            if it % 10 == 0 or True:
                tqdm.write(
                    "iter=%4d, img_loss=%.4f, reg_loss=%.4f"
                    % (it, img_loss.item() if hasattr(img_loss, 'item') else img_loss, 
                       reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss)
                )
            
            # Backward pass (same as original)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Best model tracking (same as original)
            cur_loss = total_loss.clone().detach().cpu().item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_loss_iter = it
                if hasattr(geometry, 'tet_v'):
                    best_v = geometry.tet_v.clone().detach()
                best_opt_imgs = out["shaded"].clone().detach()
            
            # Periodic saves and visualization (same as original)
            if it % 100 == 0 and forw_id == 0:
                os.makedirs(f"{cfg.output_path}/mesh{it:05d}", exist_ok=True)
                geometry.export(f"{cfg.output_path}/mesh{it:05d}", f"{it:05d}")
                
                if verbose:
                    # Save rendered image
                    opt_img = out["shaded"].clone().detach()
                    img = opt_img.cpu().numpy()
                    
                    if len(img.shape) == 3 and img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)
                    elif len(img.shape) == 3 and img.shape[2] == 4:
                        pass  # Already RGBA
                    
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_ours-{}.png".format(cfg.output_path, it))
                    
                    # Save ground truth
                    img = color_ref[0].cpu().numpy()
                    if len(img.shape) == 3 and img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)
                    
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_gt-{}.png".format(cfg.output_path, it))
                    
                    # Save difference for geometry stage
                    if cfg.get("fitting_stage", None) == "geometry":
                        diff = color_ref[0].cpu().numpy()[..., -1:] - opt_img.cpu().numpy()[..., -1:]
                        img = np.abs(diff)
                        if img.shape[2] == 1:
                            img = np.concatenate([img, img, img, img], axis=2)
                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save("{}/a_diff-{}.png".format(cfg.output_path, it))
    
    print(f"Best rendering loss: {best_loss} at iteration {best_loss_iter}")
    
    # Final export (same as original)
    geometry.export(f"{cfg.output_path}/final", "final", save_npy=True)
    
    if material is not None:
        material.export(f"{cfg.output_path}/final", "material")
        if hasattr(renderer, 'export'):
            renderer.export(f"{cfg.output_path}/final", "material")
    
    return {
        "best_loss": best_loss,
        "best_loss_iter": best_loss_iter,
        "final_geometry": geometry
    }


def create_warp_config(output_path: str = "config/warp_tetsplat_example.yaml"):
    """Create example configuration for Warp TetSplatting"""
    config = {
        # Output settings
        'output_path': 'results/warp_tetsplat',
        'verbose': True,
        
        # Training settings
        'total_num_iter': 2000,
        'fitting_stage': 'geometry',  # or 'texture'
        
        # Geometry settings
        'geometry_type': 'tetmesh_geometry',
        'geometry': {
            'tet_bbox': [[-1, -1, -1], [1, 1, 1]],
            'init_spheres_path': '../mesh_data/s.1.obj',  # Use paper's template
            'n_init_spheres': 1,
            'subdivide_depth': 2,
            'position_noise_deg': 0.0,
            'sphere_pertube_dev': 0.0,
            'optimize_geo': True,
        },
        
        # Renderer settings
        'renderer': {
            'context_type': 'cuda',
            'is_orthographic': False
        },
        
        # Data settings - Wonder3D format
        'dataloader_type': 'wonder3d',
        'data': {
            'batch_size': 1,
            'total_num_iter': 2000,
            'world_size': 1,
            'rank': 0,
            'camera_mvp_root': 'data/wonder3d/cameras',
            'camera_views': ['000', '045', '090', '135', '180', '225', '270', '315'],
            'image_root': 'data/wonder3d/images'
        },
        
        # Optimizer settings (same as original)
        'optimizer': {
            'lr': 0.01,
            'betas': [0.9, 0.99],
            'eps': 1e-8
        },
        
        # Permute surface settings
        'use_permute_surface_v': True,
        'permute_surface_v_param': {
            'start_iter': 0,
            'end_iter': 1000,
            'start_val': 0.01,
            'end_val': 0.001,
            'freq': 100
        },
        
        # Loss settings  
        'fit_depth': False,
        'fit_depth_starting_iter': 1000,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Example config saved to: {output_path}")
    return output_path


def create_blender_config(output_path: str = "config/warp_blender_example.yaml"):
    """Create Blender dataset configuration"""
    config = {
        # Output settings
        'output_path': 'results/warp_blender',
        'verbose': True,
        
        # Training settings
        'total_num_iter': 1000,
        'fitting_stage': 'geometry',
        
        # Geometry settings
        'geometry': {
            'init_spheres_path': '../mesh_data/s.1.obj',
            'optimize_geo': True,
        },
        
        # Renderer settings
        'renderer': {
            'context_type': 'cpu',  # Use CPU for compatibility
            'is_orthographic': False
        },
        
        # Data settings - Blender format
        'dataloader_type': 'blender',
        'data': {
            'batch_size': 1,
            'total_num_iter': 1000,
            'world_size': 1,
            'rank': 0,
            'image_root': 'data/blender_scene',  # Should contain transforms.json
            'resolution': 512
        },
        
        # Optimizer settings
        'optimizer': {
            'lr': 0.01,
            'betas': [0.9, 0.99],
            'eps': 1e-8
        },
        
        # Permute surface settings
        'use_permute_surface_v': False,
        
        # Loss settings  
        'fit_depth': False,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Blender config saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Warp-based TetSplatting trainer")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--create-example-config", action="store_true",
                        help="Create Wonder3D example configuration file")
    parser.add_argument("--create-blender-config", action="store_true",
                        help="Create Blender example configuration file")
    
    args, extras = parser.parse_known_args()
    
    if args.create_example_config:
        create_warp_config()
        print("Wonder3D example config created!")
        print("Usage: python warp_tetsplat_trainer.py --config config/warp_tetsplat_example.yaml")
    elif args.create_blender_config:
        create_blender_config()
        print("Blender example config created!")
        print("Usage: python warp_tetsplat_trainer.py --config config/warp_blender_example.yaml")
    elif args.config:
        cfg = load_config(args.config, cli_args=extras)
        print("Starting Warp TetSplatting training...")
        results = train_warp_tetsplat(cfg)
        print("Training completed!")
        print(f"Best loss: {results['best_loss']} at iteration {results['best_loss_iter']}")
    else:
        print("Complete Warp TetSplatting Trainer")
        print("Usage:")
        print("  python warp_tetsplat_trainer.py --create-example-config")
        print("  python warp_tetsplat_trainer.py --create-blender-config")  
        print("  python warp_tetsplat_trainer.py --config <config_file>")
        print()
        print("Available dataset types: wonder3d, mitsuba, blender")