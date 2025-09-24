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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Conditional import of materials only when needed to avoid nvdiffrast dependency
try:
    from materials import load_material
    MATERIALS_AVAILABLE = True
except ImportError:
    MATERIALS_AVAILABLE = False
    print("Warning: Materials not available (nvdiffrast not installed)")

from utils.config import load_config
from utils.optimizer import AdamUniform

# Skip original renderer import to avoid pypgo dependency
# from renderers import MeshRasterizer
from warp_mesh_rasterizer import WarpMeshRasterizer

from warp_dataloader import WarpDataLoader, load_warp_dataloader
from warp_image_loss import WarpImageLoss
from warp_tetmesh import WarpTetMesh, load_tetmesh_from_surface_mesh
from warp_geometry import load_warp_geometry


class TetMeshGeometryForwardData:
    def __init__(self, tet_v, tet_elem, surface_vid, surface_f, smooth_barrier_energy=None):
        self.tet_v = tet_v
        self.tet_elem = tet_elem
        self.v_pos = tet_v[surface_vid] if surface_vid is not None else tet_v
        self.t_pos_idx = surface_f
        self.smooth_barrier_energy = smooth_barrier_energy


class LinearInterpolateScheduler:
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


class WarpTetMeshGeometry(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.optimize_geo = True
        self.output_path = cfg.get('output_path', 'results')
        
        template_mesh_path = cfg.get('template_surface_sphere_path', 'mesh_data/s.1.obj')
        key_points_path = cfg.get('key_points_file_path', '')
        
        if os.path.exists(template_mesh_path):
            self.tetmesh = load_tetmesh_from_surface_mesh(template_mesh_path)
        else:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            self.tetmesh = load_tetmesh_from_surface_mesh(sphere)
        
        self.sphere_centers = None
        self.sphere_radii = None
        if key_points_path and os.path.exists(key_points_path):
            self._load_key_points(key_points_path)
        
        # Store smooth barrier parameters from config
        smooth_barrier_param = cfg.get('smooth_barrier_param', {})
        self.smooth_eng_coeff = smooth_barrier_param.get('smooth_eng_coeff', 2e-4)
        self.barrier_coeff = smooth_barrier_param.get('barrier_coeff', 2e-4)
        self.increase_order_iter = smooth_barrier_param.get('increase_order_iter', 1000)
        
        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tet_v = torch.from_numpy(self.tetmesh.vertices).float().to(device)
        tet_v.requires_grad_(True)
        
        # Register as parameter so optimizer can find it
        self.tet_v = torch.nn.Parameter(tet_v)
        
        self.tet_elements = torch.from_numpy(self.tetmesh.elements).long().to(device)
        self.rest_matrices = torch.from_numpy(self.tetmesh.rest_matrices).float().to(device)
        self.surface_vid = torch.from_numpy(self.tetmesh.surface_vertices).long().to(device)
        self.surface_fid = torch.from_numpy(self.tetmesh.surface_faces).long().to(device)
        
        # For compatibility with original renderer
        self.ones_surface_v = torch.ones([self.surface_vid.shape[0], 1]).to(device)
        self.zeros_surface_v = torch.zeros([self.surface_vid.shape[0], 1]).to(device)
        
        print(f"WarpTetMeshGeometry initialized: {len(self.tetmesh.vertices)} vertices, {len(self.tetmesh.elements)} tetrahedra")
    
    def _load_key_points(self, key_points_path):
        import json
        with open(key_points_path, 'r') as f:
            data = json.load(f)
        
        self.sphere_centers = np.array(data['pt'])
        self.sphere_radii = np.array(data['r'])
        print(f"Loaded {len(self.sphere_centers)} key point spheres from {key_points_path}")
    
    def __call__(self, **kwargs):
        from warp_tetmesh import WarpBarrierEnergy
        
        iter_num = kwargs.get('iter_num', 0)
        barrier_order = 3 if iter_num > self.increase_order_iter else 2
        
        barrier_energy = WarpBarrierEnergy.apply(
            self.tet_v,
            self.tet_elements.flatten().int(),
            self.rest_matrices,
            self.smooth_eng_coeff,  # Use config value
            self.barrier_coeff,     # Use config value
            barrier_order           # Increase order after specified iteration
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
    """Main training function - supports official config format"""
    verbose = cfg.get("verbose", False)
    # Setup material
    material = None
    os.makedirs(os.path.join(cfg.output_path, "final/"), exist_ok=True)
    
    # Loss function setup - match original trainer exactly  
    shade_loss = torch.nn.MSELoss()
    if cfg.get("fitting_stage", None) == "texture":
        if not MATERIALS_AVAILABLE:
            raise ImportError("Materials not available for texture fitting stage. Please install nvdiffrast.")
        assert cfg.get("material", None) is not None
        material = load_material(cfg.material_type)(cfg.material)
        shade_loss = torch.nn.L1Loss()
    
    # Load geometry - use dynamic loading to match original trainer exactly
    geometry_class = load_warp_geometry(cfg.geometry_type)
    geometry = geometry_class(cfg.geometry)
    
    # Use Warp renderer to avoid pypgo dependency
    renderer = WarpMeshRasterizer(geometry, material, cfg.renderer)
    
    # Use Warp dataloader with official config format
    # Map official dataloader names to Warp implementations with fallback
    dataloader_mapping = {
        "MistubaImgDataLoader": "mitsuba",
        "Wonder3DDataLoader": "wonder3d", 
        "BlenderDataLoader": "blender",
        "GSO_DataLoader": "gso",
        "NeRFDataLoader": "nerf"
    }
    
    # Try mapped name first, then direct name, then default fallback
    warp_dataloader_type = dataloader_mapping.get(cfg.dataloader_type, cfg.dataloader_type)
    
    try:
        dataset_class = load_warp_dataloader(warp_dataloader_type)
    except (NotImplementedError, ImportError) as e:
        print(f"Warning: Warp dataloader '{warp_dataloader_type}' not found, trying lowercase...")
        try:
            dataset_class = load_warp_dataloader(cfg.dataloader_type.lower())
        except:
            print(f"Error: No Warp implementation found for dataloader '{cfg.dataloader_type}'")
            print("Available Warp dataloaders:", list(dataloader_mapping.values()))
            raise NotImplementedError(f"Unsupported dataloader type: {cfg.dataloader_type}")
    
    dataloader = dataset_class(cfg.data)  # cfg.data contains all dataloader config
    
    num_forward_per_iter = dataloader.num_forward_per_iter
    
    # Optimizer - use AdamUniform to match original trainer
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
            
            # Compute losses - match original trainer exactly
            if cfg.get("fitting_stage", None) == "geometry":
                # Geometry fitting: MSE loss on alpha channel (like original)
                img_loss = shade_loss(out["shaded"][..., -1], color_ref[..., -1])
            else:
                # Texture fitting: L1 loss on RGB channels
                img_loss = shade_loss(out["shaded"][..., :3], color_ref[..., :3])
            
            img_loss *= 20  # Match original scaling
            
            # Depth loss (same as original)
            if fit_depth:
                depth_loss = torch.nn.MSELoss()(out["d"][..., -1], batch["d"][..., -1])
                img_loss += depth_loss
            
            # Regularization loss
            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out["geo_regularization"]
            
            loss = img_loss * 100 + reg_loss
            
            # Logging (same format as original)
            if True:  # it % 100 == 0:
                tqdm.write(
                    "iter=%4d, img_loss=%.4f, reg_loss=%.4f"
                    % (it, img_loss, reg_loss)
                )
            
            # Backward pass (same as original)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Best model tracking (same as original)
            cur_loss = loss.clone().detach().cpu().item()
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
                    chosen_idx = np.random.randint(0, batch["img"].shape[0])
                    opt_img = out["shaded"][chosen_idx].clone().detach()
                    # Save images
                    img = opt_img.cpu().numpy()

                    print(img.shape)
                    if img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)

                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_ours-{}.png".format(cfg.output_path, it))

                    img = color_ref[chosen_idx].cpu().numpy()
                    print(img.shape)
                    if img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)

                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_gt-{}.png".format(cfg.output_path, it))

                    if cfg.get("fitting_stage", None) == "geometry":
                        diff = color_ref[chosen_idx].cpu().numpy()[..., -1:] - opt_img.cpu().numpy()[..., -1:]
                        # Save images
                        img = np.abs(diff)
                        print(img.shape)
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





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp-based TetSplatting trainer - supports official config format")
    parser.add_argument("--config", type=str, required=True, help="Path to official config file (e.g., img_to_3D.yaml)")
    
    args, extras = parser.parse_known_args()
    
    cfg = load_config(args.config, cli_args=extras)
    print("Starting Warp TetSplatting training with official config...")
    print(f"Config: {args.config}")
    print(f"Experiment: {cfg.expr_name}")
    print(f"Geometry type: {cfg.geometry_type}")
    print(f"Dataloader type: {cfg.dataloader_type}")
    
    results = train_warp_tetsplat(cfg)
    print("Training completed!")
    print(f"Best loss: {results['best_loss']} at iteration {results['best_loss_iter']}")