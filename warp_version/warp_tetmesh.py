"""
NVIDIA Warp implementation of tetrahedral mesh geometry for shape fitting.
Replaces the CUDA extension with Warp kernels for easier development and maintenance.
"""

import numpy as np
import warp as wp
from typing import Optional
import trimesh


@wp.kernel
def compute_tet_deformation_gradient(
    tet_vertices: wp.array(dtype=wp.vec3),
    tet_elements: wp.array(dtype=wp.int32),
    rest_matrices: wp.array(dtype=wp.mat33),
    deformation_gradients: wp.array(dtype=wp.mat33)
):
    """Compute deformation gradient for each tetrahedron"""
    tid = wp.tid()
    
    # Get tetrahedron vertices
    v0_idx = tet_elements[tid * 4 + 0]
    v1_idx = tet_elements[tid * 4 + 1] 
    v2_idx = tet_elements[tid * 4 + 2]
    v3_idx = tet_elements[tid * 4 + 3]
    
    v0 = tet_vertices[v0_idx]
    v1 = tet_vertices[v1_idx]
    v2 = tet_vertices[v2_idx] 
    v3 = tet_vertices[v3_idx]
    
    # Current edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    
    # Current deformation matrix
    current_mat = wp.mat33(e1[0], e2[0], e3[0],
                          e1[1], e2[1], e3[1],
                          e1[2], e2[2], e3[2])
    
    # Deformation gradient F = current * rest^(-1)
    rest_inv = rest_matrices[tid]
    F = current_mat * rest_inv
    deformation_gradients[tid] = F


@wp.kernel  
def compute_barrier_energy_forward(
    deformation_gradients: wp.array(dtype=wp.mat33),
    energies: wp.array(dtype=wp.float32),
    order: wp.int32
):
    """Compute barrier energy for each tetrahedron"""
    tid = wp.tid()
    
    F = deformation_gradients[tid]
    J = wp.determinant(F)
    
    # Barrier energy: penalize negative jacobian
    if J < 0.0:
        if order == 2:
            energies[tid] = J * J 
        elif order == 4:
            energies[tid] = J * J * J * J
        else:
            energies[tid] = J * J
    else:
        energies[tid] = 0.0


@wp.kernel
def compute_barrier_energy_backward(
    tet_vertices: wp.array(dtype=wp.vec3),
    tet_elements: wp.array(dtype=wp.int32),
    rest_matrices: wp.array(dtype=wp.mat33),
    deformation_gradients: wp.array(dtype=wp.mat33),
    grad_output: wp.array(dtype=wp.float32),
    grad_vertices: wp.array(dtype=wp.vec3),
    order: wp.int32
):
    """Compute gradients of barrier energy w.r.t. vertices"""
    tid = wp.tid()
    
    F = deformation_gradients[tid]
    J = wp.determinant(F)
    
    if J >= 0.0:
        return
        
    # Get tetrahedron vertices indices
    v0_idx = tet_elements[tid * 4 + 0]
    v1_idx = tet_elements[tid * 4 + 1]
    v2_idx = tet_elements[tid * 4 + 2] 
    v3_idx = tet_elements[tid * 4 + 3]
    
    # Compute gradient of energy w.r.t. deformation gradient
    if order == 2:
        dE_dJ = 2.0 * J
    elif order == 4:
        dE_dJ = 4.0 * J * J * J
    else:
        dE_dJ = 2.0 * J
        
    dE_dJ *= grad_output[tid]
    
    # Gradient of determinant w.r.t. deformation gradient
    F_inv_T = wp.transpose(wp.inverse(F))
    dJ_dF = F_inv_T
    
    dE_dF = dE_dJ * dJ_dF
    
    # Gradient w.r.t. current edge vectors  
    rest_inv = rest_matrices[tid]
    dF_dEdges = rest_inv
    
    dE_dEdges = dE_dF * dF_dEdges
    
    # Distribute gradients to vertices (v1-v0, v2-v0, v3-v0)
    grad_e1 = wp.vec3(dE_dEdges[0, 0], dE_dEdges[1, 0], dE_dEdges[2, 0])
    grad_e2 = wp.vec3(dE_dEdges[0, 1], dE_dEdges[1, 1], dE_dEdges[2, 1]) 
    grad_e3 = wp.vec3(dE_dEdges[0, 2], dE_dEdges[1, 2], dE_dEdges[2, 2])
    
    # Accumulate gradients
    wp.atomic_add(grad_vertices, v0_idx, -(grad_e1 + grad_e2 + grad_e3))
    wp.atomic_add(grad_vertices, v1_idx, grad_e1)
    wp.atomic_add(grad_vertices, v2_idx, grad_e2)
    wp.atomic_add(grad_vertices, v3_idx, grad_e3)


class WarpTetMesh:
    """Warp-based tetrahedral mesh for shape fitting"""
    
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, 
                 surface_vertices: Optional[np.ndarray] = None,
                 surface_faces: Optional[np.ndarray] = None):
        """
        Initialize tetrahedral mesh
        
        Args:
            vertices: (N, 3) vertex positions
            elements: (M, 4) tetrahedral elements 
            surface_vertices: (S,) indices of surface vertices
            surface_faces: (F, 3) surface triangle faces
        """
        self.vertices = vertices.copy()
        self.elements = elements.copy()
        
        # Extract surface if not provided
        if surface_vertices is None or surface_faces is None:
            self._extract_surface()
        else:
            self.surface_vertices = surface_vertices
            self.surface_faces = surface_faces
            
        # Compute rest-state matrices for deformation gradient
        self._compute_rest_matrices()
        
    def _extract_surface(self):
        """Extract surface vertices and faces from tetrahedral mesh - matches original implementation"""
        # Generate all faces from tetrahedra (4 faces per tet)
        org_triangles = np.vstack([
            self.elements[:, [1, 2, 3]],
            self.elements[:, [0, 3, 2]],
            self.elements[:, [0, 1, 3]],
            self.elements[:, [0, 2, 1]],
        ])
        
        # Sort each triangle's vertices to avoid duplicates due to ordering
        triangles = np.sort(org_triangles, axis=1)
        
        # Find unique triangles and their counts
        unique_triangles, tri_idx, counts = np.unique(
            triangles, axis=0, return_index=True, return_counts=True
        )
        
        # Surface triangles appear only once (not shared between tets)
        once_tri_id = counts == 1
        surface_triangles = unique_triangles[once_tri_id]
        
        # Get unique surface vertices (global indices)
        surface_vertices = np.unique(surface_triangles)
        
        # CRITICAL: Map global vertex indices to local surface indices
        # This prevents index out of bounds errors during rendering
        vertex_mapping = {vertex_id: i for i, vertex_id in enumerate(surface_vertices)}
        
        # Remap triangle indices to local surface vertex indices
        original_surface_triangles = org_triangles[tri_idx][once_tri_id]
        mapped_triangles = np.vectorize(vertex_mapping.get)(original_surface_triangles)
        
        self.surface_vertices = surface_vertices.astype(np.int32)
        self.surface_faces = mapped_triangles.astype(np.int32)
        
    def _compute_rest_matrices(self):
        """Compute rest-state deformation matrices"""
        num_tets = len(self.elements)
        self.rest_matrices = np.zeros((num_tets, 3, 3), dtype=np.float32)
        
        for i, tet in enumerate(self.elements):
            v0, v1, v2, v3 = self.vertices[tet]
            
            # Rest edge vectors
            e1 = v1 - v0
            e2 = v2 - v0  
            e3 = v3 - v0
            
            # Rest matrix
            rest_mat = np.column_stack([e1, e2, e3])
            
            # Store inverse for deformation gradient computation
            try:
                self.rest_matrices[i] = np.linalg.inv(rest_mat)
            except np.linalg.LinAlgError:
                # Handle degenerate tetrahedra
                self.rest_matrices[i] = np.eye(3)
                
    def get_surface_mesh(self) -> trimesh.Trimesh:
        """Get surface mesh as trimesh object"""
        # Get surface vertices using global indices
        surface_verts = self.vertices[self.surface_vertices]
        
        # surface_faces already contains local indices (0 to len(surface_vertices)-1)
        # No remapping needed since _extract_surface() already did it
        return trimesh.Trimesh(vertices=surface_verts, faces=self.surface_faces)
        
    def export_mesh(self, filepath: str):
        """Export surface mesh to file"""
        mesh = self.get_surface_mesh()
        mesh.export(filepath)


def load_tetmesh_from_veg(veg_path: str) -> WarpTetMesh:
    """Load tetrahedral mesh from .veg file or generate from surface mesh"""
    if veg_path.endswith('.obj'):
        # 使用论文源码的方法：从表面网格生成四面体网格
        return load_tetmesh_from_surface_mesh(veg_path)
    elif veg_path.endswith('.veg'):
        raise NotImplementedError("VEG file loading not implemented yet")

def load_tetmesh_from_surface_mesh(obj_path: str) -> WarpTetMesh:

    import tetgen
    
    # 加载表面网格
    surface_mesh = trimesh.load(obj_path)
    print(f"Loaded surface mesh: {len(surface_mesh.vertices)} vertices, {len(surface_mesh.faces)} faces")
    
    # 使用tetgen生成四面体网格
    tet = tetgen.TetGen(surface_mesh.vertices, surface_mesh.faces)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    
    print(f"Generated tetmesh: {len(tet.node)} vertices, {len(tet.elem)} tetrahedra")
    
    return WarpTetMesh(tet.node.astype(np.float32), tet.elem.astype(np.int32))

if __name__ == "__main__":
    wp.init()
    
    tetmesh = load_tetmesh_from_veg("")
    print(f"Created tetmesh with {len(tetmesh.vertices)} vertices and {len(tetmesh.elements)} tetrahedra")
    print(f"Surface has {len(tetmesh.surface_vertices)} vertices and {len(tetmesh.surface_faces)} faces")