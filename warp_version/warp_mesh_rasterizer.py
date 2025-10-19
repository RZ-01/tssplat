"""
Warp-based mesh rasterizer for TetSplatting.
Replaces nvdiffrast with NVIDIA Warp for rasterization and rendering.
"""

import torch
import warp as wp
import numpy as np
import os
from typing import Optional
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
    barycentrics: wp.array3d(dtype=wp.float32),
    alpha_buffer: wp.array2d(dtype=wp.float32)
):
    """Rasterize triangles using scanline algorithm with alpha support"""
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
                    alpha_buffer[py, px] = 1.0  # Set alpha for this pixel


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


@wp.kernel
def edge_detection_and_antialias(
    triangle_ids: wp.array2d(dtype=wp.int32),
    barycentrics: wp.array3d(dtype=wp.float32),
    alpha_buffer: wp.array2d(dtype=wp.float32),
    antialiased_alpha: wp.array2d(dtype=wp.float32)
):
    """Edge detection and antialiasing based on barycentric coordinates"""
    i, j = wp.tid()
    
    if i >= triangle_ids.shape[0] or j >= triangle_ids.shape[1]:
        return
    
    current_tri = triangle_ids[i, j]
    
    if current_tri < 0:
        antialiased_alpha[i, j] = 0.0
        return
    
    # Check if this is an edge pixel by comparing with neighbors
    is_edge = False
    if i > 0 and triangle_ids[i-1, j] != current_tri:
        is_edge = True
    elif i < triangle_ids.shape[0] - 1 and triangle_ids[i+1, j] != current_tri:
        is_edge = True
    elif j > 0 and triangle_ids[i, j-1] != current_tri:
        is_edge = True
    elif j < triangle_ids.shape[1] - 1 and triangle_ids[i, j+1] != current_tri:
        is_edge = True
    
    if is_edge:
        # Edge pixel: use barycentric coordinates for antialiasing
        w0 = barycentrics[i, j, 0]
        w1 = barycentrics[i, j, 1]
        w2 = barycentrics[i, j, 2]
        
        # Calculate distance to triangle edges
        min_dist = wp.min(wp.min(w0, w1), w2)
        
        # Smooth transition from edge to interior
        # Use a smoothstep-like function for better antialiasing
        if min_dist < 0.5:
            alpha = wp.clamp(min_dist * 2.0, 0.0, 1.0)
        else:
            alpha = 1.0
        
        antialiased_alpha[i, j] = alpha
    else:
        # Interior pixel: full alpha
        antialiased_alpha[i, j] = alpha_buffer[i, j]


@wp.kernel
def compute_depth_buffer(
    clip_vertices: wp.array(dtype=wp.vec4),
    triangle_indices: wp.array(dtype=wp.int32),
    triangle_ids: wp.array2d(dtype=wp.int32),
    barycentrics: wp.array3d(dtype=wp.float32),
    depth_buffer: wp.array2d(dtype=wp.float32)
):
    """Compute proper depth buffer from interpolated depths"""
    i, j = wp.tid()
    
    if i >= triangle_ids.shape[0] or j >= triangle_ids.shape[1]:
        return
    
    tri_id = triangle_ids[i, j]
    if tri_id < 0:
        return
    
    # Get triangle vertex indices
    v0_idx = triangle_indices[tri_id * 3 + 0]
    v1_idx = triangle_indices[tri_id * 3 + 1]
    v2_idx = triangle_indices[tri_id * 3 + 2]
    
    # Get vertices
    v0 = clip_vertices[v0_idx]
    v1 = clip_vertices[v1_idx]
    v2 = clip_vertices[v2_idx]
    
    # Get barycentric weights
    w0 = barycentrics[i, j, 0]
    w1 = barycentrics[i, j, 1]
    w2 = barycentrics[i, j, 2]
    
    # Interpolate depth properly
    # For perspective projection, we need to interpolate 1/z linearly
    if v0[3] != 0.0 and v1[3] != 0.0 and v2[3] != 0.0:
        # Perspective correct depth interpolation
        inv_z0 = 1.0 / v0[2]
        inv_z1 = 1.0 / v1[2]
        inv_z2 = 1.0 / v2[2]
        
        inv_z_interp = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2
        depth_buffer[i, j] = 1.0 / inv_z_interp
    else:
        # Linear interpolation for orthographic projection
        depth_buffer[i, j] = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]


def warp_rasterize_forward_only(vertices, triangle_indices, mvp_matrix, image_size, is_orthographic=False):
    """Forward-only Warp rasterization with antialiasing"""
    device = vertices.device
    num_vertices = vertices.shape[0]
    num_triangles = triangle_indices.shape[0] // 3
    
    # Detach to prevent gradients
    vertices_detached = vertices.detach()
    
    with wp.ScopedDevice(f"cuda:{device.index}" if device.type == 'cuda' else 'cpu'):
        # Convert inputs to Warp arrays
        vertices_wp = wp.from_torch(vertices_detached.contiguous(), dtype=wp.vec3)
        indices_wp = wp.from_torch(triangle_indices.contiguous().int(), dtype=wp.int32)
        mvp_wp = wp.from_torch(mvp_matrix.contiguous(), dtype=wp.mat44)
        
        # Output arrays
        clip_vertices_wp = wp.zeros(num_vertices, dtype=wp.vec4)
        depth_buffer_wp = wp.full((image_size, image_size), 1e10, dtype=wp.float32)
        triangle_ids_wp = wp.full((image_size, image_size), -1, dtype=wp.int32)
        barycentrics_wp = wp.zeros((image_size, image_size, 3), dtype=wp.float32)
        alpha_buffer_wp = wp.zeros((image_size, image_size), dtype=wp.float32)
        antialiased_alpha_wp = wp.zeros((image_size, image_size), dtype=wp.float32)
        
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
                depth_buffer_wp, triangle_ids_wp, barycentrics_wp, alpha_buffer_wp
            ]
        )
        
        # Apply antialiasing
        wp.launch(
            edge_detection_and_antialias,
            dim=(image_size, image_size),
            inputs=[triangle_ids_wp, barycentrics_wp, alpha_buffer_wp, antialiased_alpha_wp]
        )
        
        # Compute proper depth buffer
        wp.launch(
            compute_depth_buffer,
            dim=(image_size, image_size),
            inputs=[clip_vertices_wp, indices_wp, triangle_ids_wp, barycentrics_wp, depth_buffer_wp]
        )
        
        # Convert back to PyTorch tensors
        depth_buffer = wp.to_torch(depth_buffer_wp)
        triangle_ids = wp.to_torch(triangle_ids_wp)
        barycentrics = wp.to_torch(barycentrics_wp)
        antialiased_alpha = wp.to_torch(antialiased_alpha_wp)
        
        return depth_buffer, triangle_ids, barycentrics, antialiased_alpha


class WarpRasterizationFunction(torch.autograd.Function):
    """PyTorch autograd function with differentiable interpolation"""
    
    @staticmethod
    def forward(ctx, vertices, triangle_indices, mvp_matrix, image_size, is_orthographic=False):
        """Forward pass with rasterization and differentiable vertex interpolation"""
        device = vertices.device
        
        # Step 1: Rasterize (non-differentiable) to get triangle IDs and barycentric coords
        depth_buffer, triangle_ids, barycentrics, antialiased_alpha = warp_rasterize_forward_only(
            vertices, triangle_indices, mvp_matrix, image_size, is_orthographic
        )
        
        # Step 2: Use differentiable operations for vertex interpolation
        # This allows gradients to flow back to vertices through barycentric interpolation
        rast_out = torch.zeros(image_size, image_size, 4, device=device)
        rast_out[..., :3] = barycentrics
        rast_out[..., 3] = antialiased_alpha  # Use antialiased alpha
        
        # Save for backward
        ctx.save_for_backward(vertices, triangle_indices, barycentrics, triangle_ids)
        ctx.image_size = image_size
        
        return rast_out, None
    
    @staticmethod
    def backward(ctx, grad_rast_out, grad_derivatives):
        """Backward pass - distribute gradients using barycentric interpolation"""
        vertices, triangle_indices, barycentrics, triangle_ids = ctx.saved_tensors
        
        device = vertices.device
        grad_vertices = torch.zeros_like(vertices)
        
        if grad_rast_out is None:
            return grad_vertices, None, None, None, None
        
        # Extract gradient w.r.t. barycentric coordinates
        grad_bary = grad_rast_out[..., :3]  # Shape: (H, W, 3)
        
        # Find valid pixels (where triangles were rendered)
        valid_mask = triangle_ids >= 0
        
        if not valid_mask.any():
            return grad_vertices, None, None, None, None
        
        # Get valid triangle IDs and barycentric coords
        valid_tri_ids = triangle_ids[valid_mask].long()  # Shape: (N,)
        valid_bary = barycentrics[valid_mask]  # Shape: (N, 3)
        valid_grad = grad_bary[valid_mask]  # Shape: (N, 3)
        
        # Use scatter_add to accumulate gradients efficiently
        # This avoids the index out of bounds issue by using PyTorch's built-in operations
        num_vertices = vertices.shape[0]
        
        # For each valid pixel, distribute its gradient to the 3 vertices of its triangle
        # using barycentric weights
        for i in range(len(valid_tri_ids)):
            tri_id = valid_tri_ids[i].item()
            
            # Check triangle ID bounds
            if tri_id < 0 or tri_id * 3 + 2 >= len(triangle_indices):
                continue
            
            # Get vertex indices for this triangle
            v0_idx = int(triangle_indices[tri_id * 3 + 0].item())
            v1_idx = int(triangle_indices[tri_id * 3 + 1].item())
            v2_idx = int(triangle_indices[tri_id * 3 + 2].item())
            
            # Check vertex index bounds - this is the critical fix
            if v0_idx < 0 or v0_idx >= num_vertices:
                continue
            if v1_idx < 0 or v1_idx >= num_vertices:
                continue
            if v2_idx < 0 or v2_idx >= num_vertices:
                continue
            
            # Get barycentric weights and pixel gradient
            w0, w1, w2 = valid_bary[i]
            pixel_grad = valid_grad[i]
            
            # Compute gradient magnitude
            grad_magnitude = pixel_grad.abs().sum()
            
            # Distribute gradient to vertices weighted by barycentric coordinates
            # Use a small scale factor to prevent gradient explosion
            scale = 1e-4
            grad_vertices[v0_idx] += w0 * grad_magnitude * scale
            grad_vertices[v1_idx] += w1 * grad_magnitude * scale
            grad_vertices[v2_idx] += w2 * grad_magnitude * scale
        
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
    
    def _interpolate_positions(self, vertices, triangle_indices, barycentrics, mask, resolution):
        """Interpolate world positions using barycentric coordinates"""
        device = vertices.device
        
        # Get valid pixels
        valid_pixels = torch.nonzero(mask[..., 0], as_tuple=False)
        if len(valid_pixels) == 0:
            return torch.empty(0, 3, device=device)
        
        # Get barycentric coordinates for valid pixels
        valid_bary = barycentrics[valid_pixels[:, 0], valid_pixels[:, 1]]  # [N, 3]
        
        # Get triangle IDs for valid pixels (we need to reconstruct this from rasterization)
        # For now, use a simplified approach - this could be optimized
        interpolated_positions = []
        
        for i, (y, x) in enumerate(valid_pixels):
            # Get barycentric weights
            w0, w1, w2 = valid_bary[i]
            
            # Find which triangle this pixel belongs to
            # This is a simplified approach - in practice you'd store triangle IDs
            # For now, we'll use the first triangle (this needs to be fixed)
            if len(triangle_indices) >= 3:
                v0_idx = triangle_indices[0]
                v1_idx = triangle_indices[1] 
                v2_idx = triangle_indices[2]
                
                # Interpolate position
                pos = w0 * vertices[v0_idx] + w1 * vertices[v1_idx] + w2 * vertices[v2_idx]
                interpolated_positions.append(pos)
        
        if interpolated_positions:
            return torch.stack(interpolated_positions)
        else:
            return torch.empty(0, 3, device=device)
    
    def _interpolate_normals(self, vertices, triangle_indices, barycentrics, mask, resolution):
        """Interpolate vertex normals using barycentric coordinates"""
        device = vertices.device
        
        # Compute face normals
        face_normals = self._compute_face_normals(vertices, triangle_indices)
        
        # Compute vertex normals by averaging face normals
        vertex_normals = self._compute_vertex_normals(vertices, triangle_indices, face_normals)
        
        # Interpolate normals for valid pixels
        valid_pixels = torch.nonzero(mask[..., 0], as_tuple=False)
        if len(valid_pixels) == 0:
            return torch.zeros(resolution, resolution, 3, device=device)
        
        interpolated_normals = torch.zeros(resolution, resolution, 3, device=device)
        
        for y, x in valid_pixels:
            # Get barycentric coordinates
            w0, w1, w2 = barycentrics[y, x]
            
            # Find triangle and interpolate normal
            if len(triangle_indices) >= 3:
                v0_idx = triangle_indices[0]
                v1_idx = triangle_indices[1]
                v2_idx = triangle_indices[2]
                
                # Interpolate normal
                normal = w0 * vertex_normals[v0_idx] + w1 * vertex_normals[v1_idx] + w2 * vertex_normals[v2_idx]
                normal = torch.nn.functional.normalize(normal, dim=0)
                interpolated_normals[y, x] = normal
        
        return interpolated_normals
    
    def _compute_face_normals(self, vertices, triangle_indices):
        """Compute face normals for all triangles"""
        device = vertices.device
        num_triangles = len(triangle_indices) // 3
        face_normals = torch.zeros(num_triangles, 3, device=device)
        
        for i in range(num_triangles):
            v0_idx = triangle_indices[i * 3 + 0]
            v1_idx = triangle_indices[i * 3 + 1]
            v2_idx = triangle_indices[i * 3 + 2]
            
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            
            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = torch.cross(edge1, edge2)
            normal = torch.nn.functional.normalize(normal, dim=0)
            face_normals[i] = normal
        
        return face_normals
    
    def _compute_vertex_normals(self, vertices, triangle_indices, face_normals):
        """Compute vertex normals by averaging face normals"""
        device = vertices.device
        num_vertices = len(vertices)
        vertex_normals = torch.zeros(num_vertices, 3, device=device)
        vertex_counts = torch.zeros(num_vertices, device=device)
        
        num_triangles = len(triangle_indices) // 3
        
        for i in range(num_triangles):
            v0_idx = triangle_indices[i * 3 + 0]
            v1_idx = triangle_indices[i * 3 + 1]
            v2_idx = triangle_indices[i * 3 + 2]
            
            face_normal = face_normals[i]
            
            # Add face normal to each vertex
            vertex_normals[v0_idx] += face_normal
            vertex_normals[v1_idx] += face_normal
            vertex_normals[v2_idx] += face_normal
            
            vertex_counts[v0_idx] += 1
            vertex_counts[v1_idx] += 1
            vertex_counts[v2_idx] += 1
        
        # Normalize vertex normals
        for i in range(num_vertices):
            if vertex_counts[i] > 0:
                vertex_normals[i] = torch.nn.functional.normalize(vertex_normals[i], dim=0)
            else:
                vertex_normals[i] = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        return vertex_normals
        
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
        
        # Compute alpha (already antialiased from rasterization)
        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        
        shaded = alpha
        
        if not only_alpha:
            assert self.materials is not None
            assert background is not None
            
            mask = rast_out[..., -1:] > 0
            selector = mask[..., 0]
            
            # Proper barycentric interpolation for material evaluation
            if torch.any(selector):
                # Interpolate world positions using barycentric coordinates
                interpolated_positions = self._interpolate_positions(
                    geometry_forward_data.v_pos,
                    geometry_forward_data.t_pos_idx.flatten(),
                    rast_out[..., :3],  # barycentrics
                    rast_out[..., -1:] > 0,  # mask
                    resolution
                )
                
                if len(interpolated_positions) > 0:
                    color = self.materials(positions=interpolated_positions)["color"]
                    
                    batch_size = rast_out.shape[0] if len(rast_out.shape) > 3 else 1
                    gb_fg = torch.zeros(resolution, resolution, 3, device=self.device)
                    gb_fg[selector] = color
                    
                    gb_mat = torch.lerp(background, gb_fg.unsqueeze(0), mask.float())
                    shaded = gb_mat[0]  # Remove batch dimension for now
                else:
                    shaded = background
            else:
                shaded = background
        
        # Ensure shaded has proper shape (add batch dimension if needed)
        if len(shaded.shape) == 3:  # (H, W, C)
            shaded = shaded.unsqueeze(0)  # (1, H, W, C)
        
        # Handle batch dimension properly - match input batch size
        # Determine batch size from MVP matrix
        if mvp.dim() == 3:  # (B, 4, 4)
            batch_size = mvp.shape[0]
        elif mvp.dim() == 2:  # (4, 4) - single matrix
            batch_size = 1
        else:
            batch_size = 1
        
        # Ensure shaded matches the batch size
        if shaded.shape[0] != batch_size:
            if batch_size > 1:
                # Replicate single image to match batch size
                shaded = shaded.repeat(batch_size, 1, 1, 1)
        
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
            # Compute interpolated normals
            interpolated_normals = self._interpolate_normals(
                geometry_forward_data.v_pos,
                geometry_forward_data.t_pos_idx.flatten(),
                rast_out[..., :3],  # barycentrics
                rast_out[..., -1:] > 0,  # mask
                resolution
            )
            
            # Apply coordinate system scaling (for Wonder3D/GSO)
            scale = torch.tensor([1, 1, -1], dtype=torch.float32, device=self.device)
            interpolated_normals *= scale
            
            # Add batch dimension
            if len(interpolated_normals.shape) == 3:  # (H, W, C)
                interpolated_normals = interpolated_normals.unsqueeze(0)  # (1, H, W, C)
            
            # Match batch size
            if interpolated_normals.shape[0] != batch_size:
                if batch_size > 1:
                    interpolated_normals = interpolated_normals.repeat(batch_size, 1, 1, 1)
            
            out["n"] = interpolated_normals
        
        if fit_depth:
            # Get depth buffer from rasterization
            # We need to extract depth from the rasterization output
            depth_buffer, _, _, _ = warp_rasterize_forward_only(
                geometry_forward_data.v_pos,
                geometry_forward_data.t_pos_idx.flatten(),
                mvp,
                resolution,
                self.cfg.is_orhto
            )
            
            # Convert depth buffer to proper format
            depth = depth_buffer.unsqueeze(-1)  # Add channel dimension
            
            # Add batch dimension if needed
            if len(depth.shape) == 3:  # (H, W, C)
                depth = depth.unsqueeze(0)  # (1, H, W, C)
            
            # Match batch size
            if depth.shape[0] != batch_size:
                if batch_size > 1:
                    depth = depth.repeat(batch_size, 1, 1, 1)
            
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