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
    tid = wp.tid()
    
    v0_idx = tet_elements[tid * 4 + 0]
    v1_idx = tet_elements[tid * 4 + 1] 
    v2_idx = tet_elements[tid * 4 + 2]
    v3_idx = tet_elements[tid * 4 + 3]
    
    v0 = tet_vertices[v0_idx]
    v1 = tet_vertices[v1_idx]
    v2 = tet_vertices[v2_idx] 
    v3 = tet_vertices[v3_idx]
    
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    
    current_mat = wp.mat33(e1[0], e2[0], e3[0],
                          e1[1], e2[1], e3[1],
                          e1[2], e2[2], e3[2])
    
    rest_inv = rest_matrices[tid]
    F = current_mat * rest_inv
    deformation_gradients[tid] = F


@wp.kernel  
def compute_barrier_energy_forward(
    deformation_gradients: wp.array(dtype=wp.mat33),
    energies: wp.array(dtype=wp.float32),
    order: wp.int32
):
    tid = wp.tid()
    
    F = deformation_gradients[tid]
    J = wp.determinant(F)
    
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
    tid = wp.tid()
    
    F = deformation_gradients[tid]
    J = wp.determinant(F)
    
    if J >= 0.0:
        return
        
    v0_idx = tet_elements[tid * 4 + 0]
    v1_idx = tet_elements[tid * 4 + 1]
    v2_idx = tet_elements[tid * 4 + 2] 
    v3_idx = tet_elements[tid * 4 + 3]
    
    if order == 2:
        dE_dJ = 2.0 * J
    elif order == 4:
        dE_dJ = 4.0 * J * J * J
    else:
        dE_dJ = 2.0 * J
        
    dE_dJ *= grad_output[tid]
    
    F_inv_T = wp.transpose(wp.inverse(F))
    dJ_dF = F_inv_T
    
    dE_dF = dE_dJ * dJ_dF
    
    rest_inv = rest_matrices[tid]
    dF_dEdges = rest_inv
    
    dE_dEdges = dE_dF * dF_dEdges
    
    grad_e1 = wp.vec3(dE_dEdges[0, 0], dE_dEdges[1, 0], dE_dEdges[2, 0])
    grad_e2 = wp.vec3(dE_dEdges[0, 1], dE_dEdges[1, 1], dE_dEdges[2, 1]) 
    grad_e3 = wp.vec3(dE_dEdges[0, 2], dE_dEdges[1, 2], dE_dEdges[2, 2])
    
    wp.atomic_add(grad_vertices, v0_idx, -(grad_e1 + grad_e2 + grad_e3))
    wp.atomic_add(grad_vertices, v1_idx, grad_e1)
    wp.atomic_add(grad_vertices, v2_idx, grad_e2)
    wp.atomic_add(grad_vertices, v3_idx, grad_e3)


class WarpTetMesh:
    
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, 
                 surface_vertices: Optional[np.ndarray] = None,
                 surface_faces: Optional[np.ndarray] = None):
        self.vertices = vertices.copy()
        self.elements = elements.copy()
        
        if surface_vertices is None or surface_faces is None:
            self._extract_surface()
        else:
            self.surface_vertices = surface_vertices
            self.surface_faces = surface_faces
            
        self._compute_rest_matrices()
        
    def _extract_surface(self):
        org_triangles = np.vstack([
            self.elements[:, [1, 2, 3]],
            self.elements[:, [0, 3, 2]],
            self.elements[:, [0, 1, 3]],
            self.elements[:, [0, 2, 1]],
        ])
        
        triangles = np.sort(org_triangles, axis=1)
        
        unique_triangles, tri_idx, counts = np.unique(
            triangles, axis=0, return_index=True, return_counts=True
        )
        
        once_tri_id = counts == 1
        surface_triangles = unique_triangles[once_tri_id]
        
        surface_vertices = np.unique(surface_triangles)
        
        vertex_mapping = {vertex_id: i for i, vertex_id in enumerate(surface_vertices)}
        
        original_surface_triangles = org_triangles[tri_idx][once_tri_id]
        mapped_triangles = np.vectorize(vertex_mapping.get)(original_surface_triangles)
        
        self.surface_vertices = surface_vertices.astype(np.int32)
        self.surface_faces = mapped_triangles.astype(np.int32)
        
    def _compute_rest_matrices(self):
        num_tets = len(self.elements)
        self.rest_matrices = np.zeros((num_tets, 3, 3), dtype=np.float32)
        
        for i, tet in enumerate(self.elements):
            v0, v1, v2, v3 = self.vertices[tet]
            
            e1 = v1 - v0
            e2 = v2 - v0  
            e3 = v3 - v0
            
            rest_mat = np.column_stack([e1, e2, e3])
            
            try:
                self.rest_matrices[i] = np.linalg.inv(rest_mat)
            except np.linalg.LinAlgError:
                self.rest_matrices[i] = np.eye(3)
                
    def get_surface_mesh(self) -> trimesh.Trimesh:
        surface_verts = self.vertices[self.surface_vertices]
        
        return trimesh.Trimesh(vertices=surface_verts, faces=self.surface_faces)
        
    def export_mesh(self, filepath: str):
        mesh = self.get_surface_mesh()
        mesh.export(filepath)


def load_tetmesh_from_veg(veg_path: str) -> WarpTetMesh:
    if veg_path.endswith('.obj'):
        return load_tetmesh_from_surface_mesh(veg_path)
    elif veg_path.endswith('.veg'):
        raise NotImplementedError("VEG file loading not implemented yet")

def load_tetmesh_from_surface_mesh(mesh_input) -> WarpTetMesh:
    import tetgen
    
    if isinstance(mesh_input, str):
        surface_mesh = trimesh.load(mesh_input)
        print(f"Loaded surface mesh from {mesh_input}: {len(surface_mesh.vertices)} vertices, {len(surface_mesh.faces)} faces")
    else:
        surface_mesh = mesh_input
        print(f"Using provided surface mesh: {len(surface_mesh.vertices)} vertices, {len(surface_mesh.faces)} faces")
    
    tet = tetgen.TetGen(surface_mesh.vertices, surface_mesh.faces)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    
    print(f"Generated tetmesh: {len(tet.node)} vertices, {len(tet.elem)} tetrahedra")
    
    return WarpTetMesh(tet.node.astype(np.float32), tet.elem.astype(np.int32))

if __name__ == "__main__":
    wp.init()
    
    tetmesh = load_tetmesh_from_veg("")
    print(f"Created tetmesh with {len(tetmesh.vertices)} vertices and {len(tetmesh.elements)} tetrahedra")
    print(f"Surface has {len(tetmesh.surface_vertices)} vertices and {len(tetmesh.surface_faces)} faces")