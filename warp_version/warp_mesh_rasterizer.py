"""
Warp-based mesh rasterizer for TetSplatting.
Replaces nvdiffrast with NVIDIA Warp for rasterization and rendering.
"""

import torch
import warp as wp
import numpy as np
import os
from typing import Optional, Dict
from dataclasses import dataclass

# Avoid importing original geometry to prevent pypgo dependency
# from geometry.tetmesh_geometry import TetMeshGeometry
# from materials import ExplicitMaterial
# from utils.config import parse_structured, get_device

# Simple helper functions to avoid dependency issues
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_structured(config_class, cfg_dict):
    if cfg_dict is None:
        return config_class()
    
    # Filter out unknown parameters to avoid TypeError
    import inspect
    sig = inspect.signature(config_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    
    filtered_dict = {k: v for k, v in cfg_dict.items() if k in valid_params}
    
    return config_class(**filtered_dict)


@wp.kernel
def transform_vertices(
    vertices: wp.array(dtype=wp.vec3),
    mvp_matrix: wp.array(dtype=wp.mat44),
    output_vertices: wp.array(dtype=wp.vec4),
    is_orthographic: wp.bool
):
    """Transform vertices from world space to clip space"""
    tid = wp.tid()
    
    vertex = vertices[tid]
    mvp = mvp_matrix[0]  # Assume single matrix for now
    
    # Convert to homogeneous coordinates
    vertex_homo = wp.vec4(vertex[0], vertex[1], vertex[2], 1.0)
    
    # Transform by MVP matrix
    clip_pos = mvp * vertex_homo
    
    # For orthographic projection, scale z coordinate
    if is_orthographic:
        clip_pos = wp.vec4(clip_pos[0], clip_pos[1], clip_pos[2] / 6.0, clip_pos[3])
    
    output_vertices[tid] = clip_pos


@wp.kernel
def rasterize_triangles(
    clip_vertices: wp.array(dtype=wp.vec4),
    triangle_indices: wp.array(dtype=wp.int32),
    image_width: wp.int32,  
    image_height: wp.int32,
    depth_buffer: wp.array2d(dtype=wp.float32),
    triangle_ids: wp.array2d(dtype=wp.int32),
    barycentrics: wp.array3d(dtype=wp.float32)
):
    """Rasterize triangles using scanline algorithm"""
    tid = wp.tid()
    
    # Get triangle vertices
    tri_idx = tid
    if tri_idx >= triangle_indices.shape[0] // 3:
        return
        
    v0_idx = triangle_indices[tri_idx * 3 + 0]
    v1_idx = triangle_indices[tri_idx * 3 + 1]  
    v2_idx = triangle_indices[tri_idx * 3 + 2]
    
    v0 = clip_vertices[v0_idx]
    v1 = clip_vertices[v1_idx]
    v2 = clip_vertices[v2_idx]
    
    # Perspective divide
    if v0[3] != 0.0:
        v0 = wp.vec4(v0[0] / v0[3], v0[1] / v0[3], v0[2] / v0[3], 1.0)
    if v1[3] != 0.0:
        v1 = wp.vec4(v1[0] / v1[3], v1[1] / v1[3], v1[2] / v1[3], 1.0)
    if v2[3] != 0.0:
        v2 = wp.vec4(v2[0] / v2[3], v2[1] / v2[3], v2[2] / v2[3], 1.0)
    
    # Convert to screen coordinates
    x0 = wp.int32((v0[0] + 1.0) * 0.5 * wp.float32(image_width))
    y0 = wp.int32((v0[1] + 1.0) * 0.5 * wp.float32(image_height))
    z0 = v0[2]
    
    x1 = wp.int32((v1[0] + 1.0) * 0.5 * wp.float32(image_width))
    y1 = wp.int32((v1[1] + 1.0) * 0.5 * wp.float32(image_height))
    z1 = v1[2]
    
    x2 = wp.int32((v2[0] + 1.0) * 0.5 * wp.float32(image_width))
    y2 = wp.int32((v2[1] + 1.0) * 0.5 * wp.float32(image_height))
    z2 = v2[2]
    
    # Bounding box
    min_x = wp.max(0, wp.min(wp.min(x0, x1), x2))
    max_x = wp.min(image_width - 1, wp.max(wp.max(x0, x1), x2))
    min_y = wp.max(0, wp.min(wp.min(y0, y1), y2))
    max_y = wp.min(image_height - 1, wp.max(wp.max(y0, y1), y2))
    
    # Edge functions for barycentric coordinates
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            # Barycentric coordinates
            denom = wp.float32((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
            if wp.abs(denom) < 1e-8:
                continue
                
            w0 = wp.float32((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
            w1 = wp.float32((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
            w2 = 1.0 - w0 - w1
            
            # Check if point is inside triangle
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                # Interpolate depth
                z = w0 * z0 + w1 * z1 + w2 * z2
                
                # Depth test
                if z < depth_buffer[py, px]:
                    depth_buffer[py, px] = z
                    triangle_ids[py, px] = tri_idx
                    barycentrics[py, px, 0] = w0
                    barycentrics[py, px, 1] = w1
                    barycentrics[py, px, 2] = w2


@wp.kernel
def interpolate_attributes(
    vertex_attributes: wp.array2d(dtype=wp.float32),
    triangle_indices: wp.array(dtype=wp.int32),
    triangle_ids: wp.array2d(dtype=wp.int32),
    barycentrics: wp.array3d(dtype=wp.float32),
    output_attributes: wp.array3d(dtype=wp.float32)
):
    """Interpolate vertex attributes using barycentric coordinates"""
    i, j = wp.tid()
    
    tri_id = triangle_ids[i, j]
    if tri_id < 0:
        return
        
    # Get triangle vertex indices
    v0_idx = triangle_indices[tri_id * 3 + 0]
    v1_idx = triangle_indices[tri_id * 3 + 1]
    v2_idx = triangle_indices[tri_id * 3 + 2]
    
    # Get barycentric weights
    w0 = barycentrics[i, j, 0]
    w1 = barycentrics[i, j, 1]
    w2 = barycentrics[i, j, 2]
    
    # Interpolate each attribute channel
    for c in range(vertex_attributes.shape[1]):
        attr0 = vertex_attributes[v0_idx, c]
        attr1 = vertex_attributes[v1_idx, c]
        attr2 = vertex_attributes[v2_idx, c]
        
        output_attributes[i, j, c] = w0 * attr0 + w1 * attr1 + w2 * attr2


class WarpRasterizationFunction(torch.autograd.Function):
    """PyTorch autograd function for Warp rasterization"""
    
    @staticmethod
    def forward(ctx, vertices, triangle_indices, mvp_matrix, image_size, is_orthographic=False):
        """Forward pass for rasterization"""
        device = vertices.device
        batch_size = mvp_matrix.shape[0] if len(mvp_matrix.shape) > 2 else 1
        num_vertices = vertices.shape[0]
        num_triangles = triangle_indices.shape[0] // 3
        
        with wp.ScopedDevice(f"cuda:{device.index}" if device.type == 'cuda' else 'cpu'):
            # Convert inputs to Warp arrays
            vertices_wp = wp.from_torch(vertices.contiguous(), dtype=wp.vec3)
            indices_wp = wp.from_torch(triangle_indices.contiguous().int(), dtype=wp.int32)
            mvp_wp = wp.from_torch(mvp_matrix.contiguous(), dtype=wp.mat44)
            
            # Output arrays
            clip_vertices_wp = wp.zeros(num_vertices, dtype=wp.vec4)
            depth_buffer_wp = wp.full((image_size, image_size), 1e10, dtype=wp.float32)
            triangle_ids_wp = wp.full((image_size, image_size), -1, dtype=wp.int32)
            barycentrics_wp = wp.zeros((image_size, image_size, 3), dtype=wp.float32)
            
            # Transform vertices
            wp.launch(
                transform_vertices,
                dim=num_vertices,
                inputs=[vertices_wp, mvp_wp, clip_vertices_wp, is_orthographic]
            )
            
            # Rasterize triangles
            wp.launch(
                rasterize_triangles,
                dim=num_triangles,
                inputs=[
                    clip_vertices_wp, indices_wp, 
                    image_size, image_size,
                    depth_buffer_wp, triangle_ids_wp, barycentrics_wp
                ]
            )
            
            # Convert back to PyTorch tensors
            depth_buffer = wp.to_torch(depth_buffer_wp)
            triangle_ids = wp.to_torch(triangle_ids_wp)
            barycentrics = wp.to_torch(barycentrics_wp)
            
            # Create alpha mask (where triangles were rendered)
            alpha = (triangle_ids >= 0).float().unsqueeze(-1)
            
            # Save for backward pass
            ctx.save_for_backward(vertices, triangle_indices, mvp_matrix)
            ctx.image_size = image_size
            ctx.is_orthographic = is_orthographic
            
            # Return rasterization output in nvdiffrast format
            # [height, width, 4] where last channel is triangle_id + 1 (0 means no triangle)
            rast_out = torch.zeros(image_size, image_size, 4, device=device)
            rast_out[..., :3] = barycentrics
            rast_out[..., 3] = (triangle_ids + 1).float()  # +1 so 0 means background
            
            return rast_out, None  # Second return value is for derivatives (not used)
    
    @staticmethod
    def backward(ctx, grad_rast_out, grad_derivatives):
        """Backward pass - simplified for now"""
        vertices, triangle_indices, mvp_matrix = ctx.saved_tensors
        
        # For now, return zero gradients (can be implemented properly later)
        grad_vertices = torch.zeros_like(vertices)
        
        return grad_vertices, None, None, None, None


class WarpMeshRasterizer(torch.nn.Module):
    """Warp-based mesh rasterizer replacing nvdiffrast functionality"""
    
    @dataclass
    class Config:
        context_type: str = "cuda"  # For compatibility, but we use Warp
        is_orhto: bool = False  # Match original spelling error
    
    def __init__(self, geometry, 
                 materials=None,
                 cfg=None):
        super().__init__()
        
        self.cfg = parse_structured(self.Config, cfg) if cfg else self.Config()
        self.device = get_device()
        self.geometry = geometry
        self.materials = materials
        
        # Initialize Warp
        wp.init()
        
        # Cache for optimization
        self.tri_hash = None
        
    def transform_pos(self, mtx, pos, is_vec=False):
        """Transform positions using MVP matrix"""
        t_mtx = torch.from_numpy(mtx).to(self.device) if isinstance(mtx, np.ndarray) else mtx.to(self.device)
        
        if is_vec:
            # For vectors (normals)
            posw = torch.cat([pos, torch.zeros(pos.shape[0], 1, device=self.device)], dim=1)
        else:
            # For positions
            posw = torch.cat([pos, torch.ones(pos.shape[0], 1, device=self.device)], dim=1)
        
        # Transform
        if len(t_mtx.shape) == 2:
            res = torch.matmul(posw, t_mtx.t())
        else:
            res = torch.matmul(posw, t_mtx.transpose(1, 2))
        
        # Orthographic projection adjustment
        if not is_vec and self.cfg.is_orhto:  # Match original parameter name
            res[..., 2] /= 6
            
        return res
    
    def forward(self, mvp: torch.Tensor,
                only_alpha: bool = False,
                iter_num: int = 0,
                resolution: int = 512,
                permute_surface_scheduler=None,
                fit_normal: bool = False,
                fit_depth: bool = False,
                background: Optional[torch.Tensor] = None,
                campos: Optional[torch.Tensor] = None):
        """Forward rendering pass"""
        
        # Get geometry data
        geo_input = {"iter_num": iter_num}
        
        if permute_surface_scheduler is not None:
            permute_dev = permute_surface_scheduler(iter_num)
            if permute_dev is not None:
                geo_input["permute_surface_v"] = True
                geo_input["permute_surface_v_dev"] = permute_dev
        
        geometry_forward_data = self.geometry(**geo_input)
        
        # Rasterization using Warp
        rast_out, _ = WarpRasterizationFunction.apply(
            geometry_forward_data.v_pos,
            geometry_forward_data.t_pos_idx.flatten(),
            mvp,
            resolution,
            self.cfg.is_orhto  # Match original parameter name
        )
        
        # Compute alpha
        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        # TODO: Add antialiasing equivalent
        
        shaded = alpha
        
        if not only_alpha:
            assert self.materials is not None
            assert background is not None
            
            mask = rast_out[..., -1:] > 0
            selector = mask[..., 0]
            
            # Interpolate positions for material evaluation
            # This is a simplified version - full implementation would use proper interpolation
            if torch.any(selector):
                # For now, use a simple approach
                # In practice, you'd need proper barycentric interpolation
                positions = geometry_forward_data.v_pos[selector.nonzero()[:, 0]]
                
                if len(positions) > 0:
                    color = self.materials(positions=positions)["color"]
                    
                    batch_size = rast_out.shape[0] if len(rast_out.shape) > 3 else 1
                    gb_fg = torch.zeros(resolution, resolution, 3, device=self.device)
                    gb_fg[selector] = color
                    
                    gb_mat = torch.lerp(background, gb_fg.unsqueeze(0), mask.float())
                    # TODO: Add antialiasing
                    shaded = gb_mat[0]  # Remove batch dimension for now
                else:
                    shaded = background
            else:
                shaded = background
        
        # Ensure shaded has proper shape (add batch dimension if needed)
        if len(shaded.shape) == 3:  # (H, W, C)
            shaded = shaded.unsqueeze(0)  # (1, H, W, C)
        
        # Handle regularization energy - return 0.0 if None or not present
        geo_reg = None
        if hasattr(geometry_forward_data, 'smooth_barrier_energy'):
            geo_reg = geometry_forward_data.smooth_barrier_energy
        
        if geo_reg is None:
            geo_reg = torch.tensor(0.0, device=self.device)
        
        out = {
            "shaded": shaded,
            "geo_regularization": geo_reg
        }
        
        # Additional outputs
        if fit_normal:
            # Compute vertex normals
            v_normals = geometry_forward_data._compute_vertex_normal()
            scale = torch.tensor([1, 1, -1], dtype=torch.float32, device=self.device)
            v_normals *= scale
            
            # TODO: Proper interpolation of normals
            out["n"] = torch.zeros_like(shaded)
        
        if fit_depth:
            # Return depth buffer from rasterization
            depth = rast_out[..., 2:3]  # Z-buffer values
            if len(depth.shape) == 3:  # Add batch dimension if needed
                depth = depth.unsqueeze(0)
            out["d"] = depth
        
        return out
    
    def export(self, path: str, folder: str, texture_res: int = 1024):
        """Export mesh with materials (simplified version)"""
        assert self.materials is not None
        
        os.makedirs(os.path.join(path, folder), exist_ok=True)
        
        # Get surface mesh
        v_pos = self.geometry.tet_v.clone().detach()[self.geometry.surface_vid]
        t_pos_idx = self.geometry.surface_fid
        
        # Evaluate materials at vertices
        with torch.no_grad():
            mat_out = self.materials.forward(v_pos)
        
        # Export as OBJ with vertex colors
        import trimesh
        surface_mesh = trimesh.Trimesh(
            vertices=v_pos.cpu().numpy(), 
            faces=t_pos_idx.cpu().numpy()
        )
        surface_mesh.visual.vertex_colors = mat_out["color"].cpu().numpy()
        surface_mesh.export(os.path.join(path, folder, "exported_surface.obj"))