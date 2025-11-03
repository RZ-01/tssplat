"""
Hybrid renderer that combines original TetSplatting materials with Warp rasterization.
Uses original material system but replaces nvdiffrast with Warp for rasterization.
"""

import torch
import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import Optional, Union
from omegaconf import DictConfig

# Import original components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from renderers.mesh_rasterizer import MeshRasterizer
from materials import ExplicitMaterial
from utils.config import parse_structured, get_device
from utils.typing import *

# Import Warp components
from warp_mesh_rasterizer import WarpMeshRasterizer, WarpRasterizationFunction
import warp as wp


@dataclass
class HybridMeshRasterizerConfig(MeshRasterizer.Config):
    """Configuration for hybrid mesh rasterizer"""
    use_warp_rasterization: bool = True  # Use Warp instead of nvdiffrast
    use_original_materials: bool = True   # Use original material system
    fallback_to_original: bool = True     # Fallback to original if Warp fails


class HybridMeshRasterizer(torch.nn.Module):
    """
    Hybrid mesh rasterizer that combines original materials with Warp rasterization.
    
    Features:
    - Uses original material system (ExplicitMaterial, etc.)
    - Uses Warp for rasterization instead of nvdiffrast
    - Maintains compatibility with original TetSplatting pipeline
    - Can fallback to original rasterization if needed
    """
    
    def __init__(self, geometry, materials: Optional[ExplicitMaterial] = None, 
                 cfg: Optional[Union[dict, DictConfig]] = None):
        super().__init__()
        
        self.cfg = parse_structured(HybridMeshRasterizerConfig, cfg)
        self.device = get_device()
        self.geometry = geometry
        self.materials = materials
        
        # Initialize Warp if using Warp rasterization
        if self.cfg.use_warp_rasterization:
            wp.init()
            self.warp_rasterizer = WarpMeshRasterizer(geometry, materials, cfg)
        
        # Initialize original rasterizer as fallback
        if self.cfg.fallback_to_original:
            try:
                self.original_rasterizer = MeshRasterizer(geometry, materials, cfg)
            except Exception as e:
                print(f"Warning: Could not initialize original rasterizer: {e}")
                self.original_rasterizer = None
    
    def forward(self, mvp: torch.Tensor,
                only_alpha: bool = False,
                iter_num: int = 0,
                resolution: int = 512,
                permute_surface_scheduler=None,
                fit_normal: bool = False,
                fit_depth: bool = False,
                background: Optional[torch.Tensor] = None,
                campos: Optional[torch.Tensor] = None):
        """
        Forward rendering pass using hybrid approach.
        
        Args:
            mvp: Model-View-Projection matrix
            only_alpha: Only render alpha channel
            iter_num: Current iteration number
            resolution: Rendering resolution
            permute_surface_scheduler: Surface permutation scheduler
            fit_normal: Whether to fit normals
            fit_depth: Whether to fit depth
            background: Background color
            campos: Camera position
            
        Returns:
            dict: Rendering results with same format as original
        """
        
        # Prepare renderer input
        renderer_input = {
            "mvp": mvp,
            "only_alpha": only_alpha,
            "iter_num": iter_num,
            "resolution": resolution,
            "permute_surface_scheduler": permute_surface_scheduler,
            "fit_normal": fit_normal,
            "fit_depth": fit_depth,
            "background": background,
            "campos": campos
        }
        
        # Try Warp rasterization first
        if self.cfg.use_warp_rasterization:
            try:
                return self._render_with_warp(**renderer_input)
            except Exception as e:
                print(f"Warning: Warp rasterization failed: {e}")
                if self.cfg.fallback_to_original and self.original_rasterizer is not None:
                    print("Falling back to original rasterization...")
                    return self._render_with_original(**renderer_input)
                else:
                    raise e
        
        # Use original rasterization
        return self._render_with_original(**renderer_input)
    
    def _render_with_warp(self, **kwargs):
        """Render using Warp rasterization"""
        return self.warp_rasterizer.forward(**kwargs)
    
    def _render_with_original(self, **kwargs):
        """Render using original nvdiffrast rasterization"""
        if self.original_rasterizer is None:
            raise RuntimeError("Original rasterizer not available")
        
        return self.original_rasterizer.forward(**kwargs)
    
    def transform_pos(self, mtx, pos, is_vec=False):
        """Transform positions using MVP matrix"""
        # Use original transformation method
        if hasattr(self, 'original_rasterizer') and self.original_rasterizer is not None:
            return self.original_rasterizer.transform_pos(mtx, pos, is_vec)
        else:
            # Fallback implementation
            t_mtx = torch.from_numpy(mtx).to(self.device) if isinstance(mtx, np.ndarray) else mtx.to(self.device)
            
            if is_vec:
                posw = torch.cat([pos, torch.zeros(pos.shape[0], 1, device=self.device)], dim=1)
            else:
                posw = torch.cat([pos, torch.ones(pos.shape[0], 1, device=self.device)], dim=1)
            
            if len(t_mtx.shape) == 2:
                res = torch.matmul(posw, t_mtx.t())
            else:
                res = torch.matmul(posw, t_mtx.transpose(1, 2))
            
            if not is_vec and self.cfg.is_orhto:
                res[..., 2] /= 6
                
            return res
    
    def export(self, path: str, folder: str, texture_res: int = 1024):
        """Export mesh with materials"""
        if self.cfg.use_warp_rasterization and hasattr(self, 'warp_rasterizer'):
            return self.warp_rasterizer.export(path, folder, texture_res)
        elif self.original_rasterizer is not None:
            return self.original_rasterizer.export(path, folder, texture_res)
        else:
            raise RuntimeError("No rasterizer available for export")


class HybridMaterialRenderer(torch.nn.Module):
    """
    Hybrid material renderer that can use different material systems.
    
    This class provides a unified interface for different material types
    while maintaining compatibility with the original pipeline.
    """
    
    def __init__(self, material_type: str, material_cfg: dict):
        super().__init__()
        self.material_type = material_type
        self.material_cfg = material_cfg
        
        # Load material based on type
        if material_type == "explicit":
            from materials import ExplicitMaterial
            self.material = ExplicitMaterial(material_cfg)
        elif material_type == "neural":
            # Add neural material support if needed
            raise NotImplementedError("Neural materials not implemented yet")
        else:
            raise ValueError(f"Unknown material type: {material_type}")
    
    def forward(self, **kwargs):
        """Forward pass for material evaluation"""
        return self.material.forward(**kwargs)
    
    def parameters(self):
        """Return material parameters for optimization"""
        return self.material.parameters()


def create_hybrid_renderer(geometry, material_type: str = None, 
                          material_cfg: dict = None, renderer_cfg: dict = None):
    """
    Create a hybrid renderer with specified configuration.
    
    Args:
        geometry: Geometry object
        material_type: Type of material to use
        material_cfg: Material configuration
        renderer_cfg: Renderer configuration
        
    Returns:
        HybridMeshRasterizer: Configured hybrid renderer
    """
    
    # Create material if specified
    materials = None
    if material_type is not None and material_cfg is not None:
        materials = HybridMaterialRenderer(material_type, material_cfg)
    
    # Create hybrid renderer
    renderer = HybridMeshRasterizer(geometry, materials, renderer_cfg)
    
    return renderer
