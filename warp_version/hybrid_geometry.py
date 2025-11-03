"""Hybrid geometry class combining original TetSplatting with Warp components."""

import torch
import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from geometry.tetmesh_geometry import TetMeshGeometry, TetMeshMultiSphereGeometry, TetMeshGeometryForwardData
from geometry.tetrahedron_mesh import TetrahedronMesh
from utils.config import parse_structured, get_device
from utils.typing import *

import warp as wp


@dataclass
class HybridTetMeshGeometryConfig(TetMeshGeometry.Config):
    """Configuration for hybrid geometry class"""
    use_warp_for_simple_ops: bool = True
    use_original_energy: bool = True
    use_warp_rasterization: bool = True


class HybridTetMeshGeometry(TetMeshGeometry):
    """Hybrid geometry combining original TetSplatting with Warp."""
    
    def __init__(self, cfg: Optional[DictConfig] = None):
        super().__init__(cfg)
        self.hybrid_cfg = parse_structured(HybridTetMeshGeometryConfig, cfg)
        
        if self.hybrid_cfg.use_warp_for_simple_ops or self.hybrid_cfg.use_warp_rasterization:
            wp.init()
            self.warp_tetmesh = None
    
    def forward(self, **kwargs):
        """Forward pass combining original and Warp components."""
        iter_num = kwargs.get("iter_num", 0)
        
        if "permute_surface_v" in kwargs and kwargs["permute_surface_v"]:
            permute_dev = kwargs.get("permute_surface_v_dev", 0.01)
            self._permute_surface_vertices(permute_dev)
        
        forward_data = self._compute_geometry_forward_data()
        
        if self.hybrid_cfg.use_warp_for_simple_ops:
            forward_data.warp_tetmesh = None
        
        return forward_data
    
    def _compute_geometry_forward_data(self):
        """Compute geometry forward data using original methods"""
        surface_v = self.tet_v[self.surface_vid]
        surface_f = self.surface_fid
        
        total_reg_energy = 0.0
        if self.mesh_smooth_barrier is not None:
            total_reg_energy = self.mesh_smooth_barrier(self.tet_v)
        
        forward_data = TetMeshGeometryForwardData(
            tet_v=self.tet_v,
            tet_elem=self.tet_elem,
            surface_vid=self.surface_vid,
            surface_f=surface_f,
            smooth_barrier_energy=total_reg_energy
        )
        
        forward_data.v_pos = surface_v
        forward_data.t_pos_idx = surface_f
        
        return forward_data
    
    def _permute_surface_vertices(self, dev: float):
        """Permute surface vertices using original method"""
        if hasattr(self, 'surface_vid'):
            noise = torch.randn_like(self.tet_v[self.surface_vid]) * dev
            self.tet_v.data[self.surface_vid] += noise
    
    def _compute_vertex_normal(self):
        """Compute vertex normals using original method"""
        return torch.zeros_like(self.tet_v[self.surface_vid])
    
    def reset(self, tet_v_np: np.array, tet_elem_np: np.array, 
              surface_vid_np: Optional[np.array] = None, 
              surface_fid_np: Optional[np.array] = None):
        """Reset geometry with new data"""
        super().reset(tet_v_np, tet_elem_np, surface_vid_np, surface_fid_np)
        
        if hasattr(self, 'warp_tetmesh'):
            self.warp_tetmesh = None


@dataclass
class HybridTetMeshMultiSphereGeometryConfig(TetMeshMultiSphereGeometry.Config):
    """Configuration for hybrid multi-sphere geometry"""
    use_warp_for_simple_ops: bool = True
    use_original_energy: bool = True
    use_warp_rasterization: bool = True


class HybridTetMeshMultiSphereGeometry(TetMeshMultiSphereGeometry):
    """Hybrid multi-sphere geometry combining original TetSplatting with Warp."""
    
    def __init__(self, cfg: Optional[DictConfig] = None):
        super().__init__(cfg)
        self.hybrid_cfg = parse_structured(HybridTetMeshMultiSphereGeometryConfig, cfg)
        
        if self.hybrid_cfg.use_warp_for_simple_ops or self.hybrid_cfg.use_warp_rasterization:
            wp.init()
    
    def forward(self, **kwargs):
        """Forward pass for multi-sphere geometry"""
        iter_num = kwargs.get("iter_num", 0)
        
        if "permute_surface_v" in kwargs and kwargs["permute_surface_v"]:
            permute_dev = kwargs.get("permute_surface_v_dev", 0.01)
            self._permute_surface_vertices(permute_dev)
        
        forward_data = self._compute_multi_sphere_forward_data(iter_num)
        return forward_data
    
    def _compute_multi_sphere_forward_data(self, iter_num: int):
        """Compute forward data for multi-sphere geometry"""
        all_surface_v = []
        all_surface_f = []
        face_offset = 0
        
        for sphere in self.tet_spheres:
            surface_v = sphere.tet_v[sphere.surface_vid]
            surface_f = sphere.surface_fid + face_offset
            
            all_surface_v.append(surface_v)
            all_surface_f.append(surface_f)
            
            face_offset += len(sphere.surface_vid)
        
        surface_v = torch.cat(all_surface_v, dim=0)
        surface_f = torch.cat(all_surface_f, dim=0)
        
        total_reg_energy = 0.0
        for sphere in self.tet_spheres:
            if sphere.mesh_smooth_barrier is not None:
                total_reg_energy += sphere.mesh_smooth_barrier(sphere.tet_v)
        
        forward_data = TetMeshGeometryForwardData(
            tet_v=surface_v,
            tet_elem=surface_f,
            surface_vid=torch.arange(len(surface_v), device=self.device),
            surface_f=surface_f,
            smooth_barrier_energy=total_reg_energy
        )
        
        forward_data.v_pos = surface_v
        forward_data.t_pos_idx = surface_f
        
        return forward_data
    
    def _permute_surface_vertices(self, dev: float):
        """Permute surface vertices for all spheres"""
        for sphere in self.tet_spheres:
            if hasattr(sphere, 'surface_vid'):
                noise = torch.randn_like(sphere.tet_v[sphere.surface_vid]) * dev
                sphere.tet_v.data[sphere.surface_vid] += noise


def load_hybrid_geometry(geometry_class_type: str):
    """Load hybrid geometry class"""
    if geometry_class_type == "HybridTetMeshGeometry":
        return HybridTetMeshGeometry
    elif geometry_class_type == "HybridTetMeshMultiSphereGeometry":
        return HybridTetMeshMultiSphereGeometry
    else:
        from geometry import load_geometry
        return load_geometry(geometry_class_type)
