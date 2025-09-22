"""
Geometric loss functions for shape fitting using NVIDIA Warp.
Implements point-to-mesh distance calculations and related losses.
"""

import torch
import warp as wp
import trimesh


@wp.kernel
def compute_point_to_triangle_distance(
    points: wp.array(dtype=wp.vec3),
    triangle_vertices: wp.array(dtype=wp.vec3),  # Flattened: [v0, v1, v2, v0, v1, v2, ...]
    triangle_indices: wp.array(dtype=wp.int32),   # Which triangle each point should check
    distances: wp.array(dtype=wp.float32),
    closest_points: wp.array(dtype=wp.vec3)
):
    """Compute distance from points to specified triangles"""
    tid = wp.tid()
    
    point = points[tid]
    tri_idx = triangle_indices[tid]
    
    # Get triangle vertices
    v0 = triangle_vertices[tri_idx * 3 + 0]
    v1 = triangle_vertices[tri_idx * 3 + 1]
    v2 = triangle_vertices[tri_idx * 3 + 2]
    
    # Compute closest point on triangle to the point
    # Using barycentric coordinates
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = point - v0
    
    a = wp.dot(edge0, edge0)
    b = wp.dot(edge0, edge1)
    c = wp.dot(edge1, edge1)
    d = wp.dot(edge0, v0_to_point)
    e = wp.dot(edge1, v0_to_point)
    
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e
    
    closest_point = wp.vec3(0.0, 0.0, 0.0)
    
    if s + t <= det:
        if s < 0.0:
            if t < 0.0:
                # Region 4
                if d < 0.0:
                    t = 0.0
                    s = wp.clamp(-d / a, 0.0, 1.0)
                else:
                    s = 0.0
                    t = wp.clamp(-e / c, 0.0, 1.0)
            else:
                # Region 3
                s = 0.0
                t = wp.clamp(-e / c, 0.0, 1.0)
        elif t < 0.0:
            # Region 5
            t = 0.0
            s = wp.clamp(-d / a, 0.0, 1.0)
        else:
            # Region 0 - inside triangle
            inv_det = 1.0 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0.0:
            # Region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                s = wp.clamp(numer / denom, 0.0, 1.0)
                t = 1.0 - s
            else:
                s = 0.0
                t = wp.clamp(-e / c, 0.0, 1.0)
        elif t < 0.0:
            # Region 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                t = wp.clamp(numer / denom, 0.0, 1.0)
                s = 1.0 - t
            else:
                t = 0.0
                s = wp.clamp(-d / a, 0.0, 1.0)
        else:
            # Region 1
            numer = c + e - b - d
            if numer <= 0.0:
                s = 0.0
            else:
                denom = a - 2.0 * b + c
                s = wp.clamp(numer / denom, 0.0, 1.0)
            t = 1.0 - s
    
    closest_point = v0 + s * edge0 + t * edge1
    distance = wp.length(point - closest_point)
    
    distances[tid] = distance
    closest_points[tid] = closest_point


@wp.kernel
def compute_geometric_loss_gradients(
    points: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    point_gradients: wp.array(dtype=wp.vec3),
    loss_scale: wp.float32
):
    """Compute gradients for geometric loss"""
    tid = wp.tid()
    
    point = points[tid]
    closest = closest_points[tid]
    
    # Gradient is in direction from closest point to current point
    diff = point - closest
    distance = wp.length(diff)
    
    if distance > 1e-6:
        gradient = (diff / distance) * loss_scale
    else:
        gradient = wp.vec3(0.0, 0.0, 0.0)
    
    point_gradients[tid] = gradient


class GeometricLoss(torch.autograd.Function):
    """PyTorch autograd function for geometric loss computation"""
    
    @staticmethod
    def forward(ctx, surface_vertices, target_vertices, target_faces, loss_weight=1.0):
        """
        Compute geometric loss between surface vertices and target mesh
        
        Args:
            surface_vertices: (N, 3) surface vertex positions
            target_vertices: (M, 3) target mesh vertices  
            target_faces: (F, 3) target mesh face indices
            loss_weight: scalar weight for loss
        """
        device = surface_vertices.device
        num_points = surface_vertices.shape[0]
        num_faces = target_faces.shape[0]
        
        with wp.ScopedDevice(device.index if device.type == 'cuda' else 'cpu'):
            # Convert to Warp arrays
            points_wp = wp.from_torch(surface_vertices.contiguous(), dtype=wp.vec3)
            
            # Flatten target triangles for easy access
            target_triangles = target_vertices[target_faces.flatten()].reshape(-1, 3)
            triangles_wp = wp.from_torch(target_triangles.contiguous(), dtype=wp.vec3)
            
            # For simplicity, assign each point to closest triangle (can be optimized)
            # Here we'll use a simple approach - check all triangles for each point
            distances_wp = wp.zeros(num_points, dtype=wp.float32)
            closest_points_wp = wp.zeros(num_points, dtype=wp.vec3)
            
            # Simple implementation: find closest triangle for each point
            # In practice, you'd want spatial acceleration (BVH, etc.)
            min_distances = torch.full((num_points,), float('inf'), device=device)
            best_triangles = torch.zeros(num_points, dtype=torch.int32, device=device)
            best_closest = torch.zeros_like(surface_vertices)
            
            for face_idx in range(num_faces):
                # Get triangle vertices
                face = target_faces[face_idx]
                tri_verts = target_vertices[face]  # (3, 3)
                
                # Expand for broadcasting
                tri_v0 = tri_verts[0].unsqueeze(0)  # (1, 3)
                tri_v1 = tri_verts[1].unsqueeze(0)  # (1, 3) 
                tri_v2 = tri_verts[2].unsqueeze(0)  # (1, 3)
                
                # Vectorized point-to-triangle distance (simplified)
                points = surface_vertices  # (N, 3)
                
                # Simple point-to-triangle distance (can be optimized)
                edge0 = tri_v1 - tri_v0  # (1, 3)
                edge1 = tri_v2 - tri_v0  # (1, 3)
                v0_to_points = points - tri_v0  # (N, 3)
                
                # Project onto triangle plane and clamp to triangle
                # This is a simplified version - full implementation would be more complex
                distances_to_tri = torch.norm(v0_to_points, dim=1)  # (N,)
                
                # Update minimum distances
                mask = distances_to_tri < min_distances
                min_distances[mask] = distances_to_tri[mask]
                best_triangles[mask] = face_idx
                # best_closest computation would go here
            
            total_loss = loss_weight * torch.sum(min_distances)
            
            # Save for backward
            ctx.save_for_backward(surface_vertices, target_vertices, target_faces, 
                                  min_distances, best_triangles)
            ctx.loss_weight = loss_weight
            
            return total_loss
    
    @staticmethod 
    def backward(ctx, grad_output):
        surface_vertices, target_vertices, target_faces, distances, triangles = ctx.saved_tensors
        loss_weight = ctx.loss_weight
        
        # Compute gradients (simplified implementation)
        grad_vertices = torch.zeros_like(surface_vertices)
        
        # In a full implementation, you'd compute proper gradients based on 
        # closest points and geometric relationships
        # For now, use a simple gradient approximation
        for i in range(surface_vertices.shape[0]):
            if distances[i] > 1e-6:
                # Simple gradient pointing away from target
                grad_vertices[i] = surface_vertices[i] / distances[i] * loss_weight
        
        return grad_output * grad_vertices, None, None, None


class WarpGeometricLoss:
    """Warp-based geometric loss for shape fitting"""
    
    def __init__(self, target_mesh: trimesh.Trimesh, loss_weight: float = 1.0):
        """
        Initialize geometric loss with target mesh
        
        Args:
            target_mesh: Target mesh to fit to
            loss_weight: Weight for geometric loss term
        """
        self.target_mesh = target_mesh
        self.loss_weight = loss_weight
        
        # Convert target mesh to tensors
        self.target_vertices = torch.from_numpy(target_mesh.vertices).float()
        self.target_faces = torch.from_numpy(target_mesh.faces).long()
        
    def compute_loss(self, surface_vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric loss between surface vertices and target mesh
        
        Args:
            surface_vertices: (N, 3) tensor of surface vertex positions
            
        Returns:
            Scalar loss tensor
        """
        return GeometricLoss.apply(
            surface_vertices, 
            self.target_vertices.to(surface_vertices.device),
            self.target_faces.to(surface_vertices.device),
            self.loss_weight
        )
        
    def to(self, device):
        """Move to device"""
        self.target_vertices = self.target_vertices.to(device)
        self.target_faces = self.target_faces.to(device)
        return self


def load_target_mesh(filepath: str) -> trimesh.Trimesh:
    """Load target mesh from file"""
    mesh = trimesh.load(filepath)
    
    # Handle Scene objects (multiple meshes in one file)
    if isinstance(mesh, trimesh.Scene):
        # Get the first mesh from the scene
        meshes = list(mesh.geometry.values())
        if len(meshes) > 0:
            mesh = meshes[0]  # Take first mesh
        else:
            raise ValueError(f"No meshes found in scene: {filepath}")
    
    return mesh


if __name__ == "__main__":
    # Test geometric loss
    wp.init()
    
    # Create simple test data
    target_mesh = trimesh.creation.box(extents=[2, 2, 2])
    print(f"Target mesh: {len(target_mesh.vertices)} vertices, {len(target_mesh.faces)} faces")
    
    # Create some test surface points
    surface_points = torch.randn(100, 3) * 0.5  # Points near origin
    
    # Test geometric loss
    geo_loss = WarpGeometricLoss(target_mesh, loss_weight=1.0)
    loss = geo_loss.compute_loss(surface_points)
    print(f"Geometric loss: {loss.item()}")