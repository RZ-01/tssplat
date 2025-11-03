"""
Hybrid trainer that combines original TetSplatting training with Warp components.
Maintains original training pipeline while using Warp for rasterization.
"""

import torch
import numpy as np
import os
import sys
from tqdm import trange
from dataclasses import dataclass
from typing import Optional, Union
from omegaconf import DictConfig

# Import original components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import parse_structured, get_device
from utils.typing import *
from utils.optimizer import AdamUniform
from utils.scheduler import CosineAnnealingLR

# Import hybrid components
from hybrid_geometry import load_hybrid_geometry, HybridTetMeshGeometry, HybridTetMeshMultiSphereGeometry
from hybrid_renderer import create_hybrid_renderer, HybridMeshRasterizer

# Import original dataloaders and materials
from data.dataloader import load_dataloader
from materials import load_material


@dataclass
class HybridTrainerConfig:
    """Configuration for hybrid trainer"""
    # Training parameters
    total_num_iter: int = 10000
    verbose: bool = False
    
    # Component selection
    use_warp_rasterization: bool = True
    use_original_geometry: bool = True
    use_original_materials: bool = True
    use_original_dataloader: bool = True
    
    # Fallback options
    fallback_to_original: bool = True
    
    # Output
    output_path: str = "results/hybrid"


def train_hybrid_tetsplat(cfg: Union[dict, DictConfig]):
    """
    Hybrid TetSplatting training that combines original and Warp components.
    
    This function maintains the original training pipeline while allowing
    selective use of Warp components for specific operations.
    
    Args:
        cfg: Configuration dictionary containing all training parameters
    """
    
    # Parse configuration
    hybrid_cfg = parse_structured(HybridTrainerConfig, cfg.get("hybrid", {}))
    
    verbose = cfg.get("verbose", False)
    material = None
    
    # Create output directory
    os.makedirs(os.path.join(cfg.output_path, "final/"), exist_ok=True)
    
    # Setup loss function
    shade_loss = torch.nn.MSELoss()
    if cfg.get("fitting_stage", None) == "texture":
        try:
            assert cfg.get("material", None) is not None
            material = load_material(cfg.material_type)(cfg.material)
            shade_loss = torch.nn.L1Loss()
        except ImportError:
            raise ImportError("Materials not available for texture fitting stage.")
    
    # Load geometry
    if hybrid_cfg.use_original_geometry:
        # Use original geometry classes
        from geometry import load_geometry
        geometry_class = load_geometry(cfg.geometry_type)
    else:
        # Use hybrid geometry classes
        geometry_class = load_hybrid_geometry(cfg.geometry_type)
    
    geometry = geometry_class(cfg.geometry)
    
    # Create renderer
    if hybrid_cfg.use_warp_rasterization:
        # Use hybrid renderer with Warp rasterization
        renderer = create_hybrid_renderer(
            geometry=geometry,
            material_type=cfg.get("material_type", "explicit"),
            material_cfg=cfg.get("material", {}),
            renderer_cfg=cfg.get("renderer", {})
        )
    else:
        # Use original renderer
        from renderers.mesh_rasterizer import MeshRasterizer
        renderer = MeshRasterizer(geometry, material, cfg.renderer)
    
    # Load dataloader (always use original dataloader)
    dataloader_class = load_dataloader(cfg.dataloader_type)
    
    dataloader = dataloader_class(cfg.data)
    num_forward_per_iter = dataloader.num_forward_per_iter
    
    # Setup optimizer
    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
    scheduler = CosineAnnealingLR(
        optimizer, cfg.total_num_iter * num_forward_per_iter, eta_min=1e-4)
    
    # Setup surface permutation scheduler
    permute_surface_scheduler = None
    if cfg.get('use_permute_surface_v', False):
        # Simple linear interpolation scheduler
        class LinearInterpolateScheduler:
            def __init__(self, start_iter, end_iter, freq, start_val, end_val):
                self.start_iter = start_iter
                self.end_iter = end_iter
                self.freq = freq
                self.start_val = start_val
                self.end_val = end_val
            
            def __call__(self, iter):
                if iter < self.start_iter:
                    return None
                if iter > self.end_iter:
                    return self.end_val
                
                p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
                return self.start_val * (1 - p) + self.end_val * p
        
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    # Training state
    best_loss = 1e10
    best_loss_iter = 0
    best_opt_imgs = None
    best_v = None
    
    print("Starting Hybrid TetSplatting training...")
    print(f"Total iterations: {cfg.total_num_iter}")
    print(f"Forward passes per iteration: {num_forward_per_iter}")
    print(f"Dataset size: {len(dataloader)}")
    print(f"Using Warp rasterization: {hybrid_cfg.use_warp_rasterization}")
    print(f"Using original geometry: {hybrid_cfg.use_original_geometry}")
    
    # Training loop
    for it in trange(cfg.total_num_iter):
        for forw_id in range(num_forward_per_iter):
            batch = dataloader(it, forw_id)
            
            color_ref = batch["img"]
            
            # Determine if we should fit depth
            fit_depth = cfg.get("fit_depth", False)
            if fit_depth:
                fit_depth = cfg.get("fit_depth_starting_iter", 0) < it
            
            # Prepare renderer input
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
            try:
                out = renderer(**renderer_input)
            except Exception as e:
                if hybrid_cfg.fallback_to_original and hybrid_cfg.use_warp_rasterization:
                    print(f"Warning: Hybrid renderer failed, falling back to original: {e}")
                    # Fallback to original renderer
                    from renderers.mesh_rasterizer import MeshRasterizer
                    original_renderer = MeshRasterizer(geometry, material, cfg.renderer)
                    out = original_renderer(**renderer_input)
                else:
                    raise e
            
            # Compute losses
            if cfg.get("fitting_stage", None) == "geometry":
                img_loss = shade_loss(out["shaded"][..., -1], color_ref[..., -1])
            else:
                img_loss = shade_loss(out["shaded"][..., :3], color_ref[..., :3])
            
            img_loss *= 20
            
            if fit_depth:
                img_loss += shade_loss(out["d"][..., -1] * color_ref[..., -1],
                                       batch["d"][..., -1] * color_ref[..., -1]) * 100
            
            # Regularization loss
            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out["geo_regularization"] * cfg.get("reg_loss_weight", 0.1)
            
            total_loss = img_loss + reg_loss
            
            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track best results
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_loss_iter = it
                best_opt_imgs = out["shaded"].detach().clone()
                if hasattr(geometry, 'tet_v'):
                    best_v = geometry.tet_v.detach().clone()
            
            # Logging
            if verbose and it % 100 == 0:
                print(f"Iteration {it}: Loss = {total_loss.item():.6f}, "
                      f"Img = {img_loss.item():.6f}, Reg = {reg_loss.item():.6f}")
    
    # Save final results
    print(f"Training completed. Best loss: {best_loss:.6f} at iteration {best_loss_iter}")
    
    if best_v is not None:
        # Save best geometry
        final_output_dir = os.path.join(cfg.output_path, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Export final mesh
        if hasattr(renderer, 'export'):
            renderer.export(final_output_dir, "final_mesh")
        
        # Save geometry parameters
        torch.save({
            'tet_v': best_v,
            'geometry_cfg': cfg.geometry,
            'best_loss': best_loss,
            'best_iter': best_loss_iter
        }, os.path.join(final_output_dir, "best_geometry.pth"))
    
    return {
        'best_loss': best_loss,
        'best_iter': best_loss_iter,
        'best_images': best_opt_imgs,
        'final_geometry': best_v
    }


def train_mesh_fitting_hybrid(cfg: Union[dict, DictConfig]):
    """
    Hybrid mesh fitting training that combines original TetSplatting with Warp.
    
    This function is specifically designed for mesh fitting tasks where
    we want to use Warp for rasterization but keep original geometry optimization.
    """
    
    # Parse configuration
    hybrid_cfg = parse_structured(HybridTrainerConfig, cfg.get("hybrid", {}))
    
    verbose = cfg.get("verbose", False)
    
    # Load target mesh if provided
    target_vertices = None
    if cfg.get("target_mesh_path"):
        import trimesh
        target_mesh = trimesh.load(cfg.target_mesh_path)
        target_vertices = torch.from_numpy(target_mesh.vertices).float().to(get_device())
    
    # Load geometry
    if hybrid_cfg.use_original_geometry:
        from geometry import load_geometry
        geometry_class = load_geometry(cfg.geometry_type)
    else:
        geometry_class = load_hybrid_geometry(cfg.geometry_type)
    
    geometry = geometry_class(cfg.geometry)
    
    # Create renderer
    if hybrid_cfg.use_warp_rasterization:
        renderer = create_hybrid_renderer(
            geometry=geometry,
            material_type=None,  # No materials for mesh fitting
            material_cfg={},
            renderer_cfg=cfg.get("renderer", {})
        )
    else:
        from renderers.mesh_rasterizer import MeshRasterizer
        renderer = MeshRasterizer(geometry, None, cfg.renderer)
    
    # Setup optimizer
    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
    scheduler = CosineAnnealingLR(
        optimizer, cfg.total_num_iter, eta_min=1e-4)
    
    # Setup surface permutation scheduler
    permute_surface_scheduler = None
    if cfg.get('use_permute_surface_v', False):
        # Simple linear interpolation scheduler
        class LinearInterpolateScheduler:
            def __init__(self, start_iter, end_iter, freq, start_val, end_val):
                self.start_iter = start_iter
                self.end_iter = end_iter
                self.freq = freq
                self.start_val = start_val
                self.end_val = end_val
            
            def __call__(self, iter):
                if iter < self.start_iter:
                    return None
                if iter > self.end_iter:
                    return self.end_val
                
                p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
                return self.start_val * (1 - p) + self.end_val * p
        
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    # Training parameters
    mesh_loss_weight = cfg.get("mesh_loss_weight", 10000)
    reg_loss_weight = cfg.get("reg_loss_weight", 0.05)
    mesh_sample_ratio = cfg.get("mesh_loss_sample_ratio", 1.0)
    loss_batch_size = cfg.get("loss_batch_size", 2000)
    
    print("Starting Hybrid Mesh Fitting training...")
    print(f"Total iterations: {cfg.total_num_iter}")
    print(f"Target mesh: {cfg.get('target_mesh_path', 'None')}")
    print(f"Using Warp rasterization: {hybrid_cfg.use_warp_rasterization}")
    
    # Training loop
    pbar = trange(cfg.total_num_iter, desc="Training")
    
    for it in pbar:
        # Prepare renderer input
        renderer_input = {
            "iter_num": it,
            "permute_surface_scheduler": permute_surface_scheduler,
        }
        
        # Forward pass
        try:
            out = renderer.compute_geometry_forward(**renderer_input)
        except AttributeError:
            # Fallback to geometry forward if renderer doesn't have compute_geometry_forward
            out = geometry(**renderer_input)
        
        # Compute mesh distance loss
        if target_vertices is not None:
            current_surface_vertices = geometry.tet_v[geometry.surface_vid]
            raw_mesh_loss = compute_mesh_distance_loss(
                current_surface_vertices, 
                target_vertices, 
                batch_size=loss_batch_size,
                sample_ratio=mesh_sample_ratio
            )
            data_loss = raw_mesh_loss * mesh_loss_weight
        else:
            data_loss = torch.tensor(0.0, device=get_device())
            raw_mesh_loss = torch.tensor(0.0, device=get_device())
        
        # Regularization loss
        reg_loss = 0.0
        if cfg.get("fitting_stage", None) == "geometry":
            reg_loss = out["geo_regularization"] * reg_loss_weight
        
        total_loss = data_loss + reg_loss
        
        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'mesh': f'{raw_mesh_loss.item():.4f}',
            'reg': f'{reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:.4f}',
        })
    
    # Save final results
    final_output_dir = os.path.join(cfg.output_path, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Export final mesh
    if hasattr(renderer, 'export'):
        renderer.export(final_output_dir, "final_mesh")
    
    # Save final geometry
    torch.save({
        'tet_v': geometry.tet_v.detach(),
        'geometry_cfg': cfg.geometry,
        'final_loss': total_loss.item()
    }, os.path.join(final_output_dir, "final_geometry.pth"))
    
    print(f"Mesh fitting completed. Final loss: {total_loss.item():.6f}")


def compute_mesh_distance_loss(source_vertices, target_vertices, batch_size=2000, sample_ratio=1.0):
    """
    Compute mesh distance loss between source and target vertices.
    
    This is a simplified implementation. In practice, you might want to use
    more sophisticated distance metrics like Chamfer distance.
    """
    device = source_vertices.device
    
    # Sample vertices if needed
    if sample_ratio < 1.0:
        num_samples = int(len(target_vertices) * sample_ratio)
        indices = torch.randperm(len(target_vertices))[:num_samples]
        target_vertices = target_vertices[indices]
    
    # Compute pairwise distances in batches
    total_loss = 0.0
    num_batches = (len(target_vertices) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(target_vertices))
        batch_target = target_vertices[start_idx:end_idx]
        
        # Compute distances from batch_target to all source_vertices
        distances = torch.cdist(batch_target, source_vertices)
        min_distances = torch.min(distances, dim=1)[0]
        
        total_loss += torch.mean(min_distances)
    
    return total_loss / num_batches
