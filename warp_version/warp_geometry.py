"""
Warp geometry loading system - provides dynamic geometry type loading
compatible with original geometry system
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from warp_tetmesh import WarpTetMesh, load_tetmesh_from_surface_mesh
import numpy as np
import json
import trimesh
import tetgen
import math
import torch


class WarpTetMeshMultiSphereGeometry(torch.nn.Module):
    """
    Warp implementation of TetMeshMultiSphereGeometry
    Fully compatible with original multi-sphere initialization logic
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.optimize_geo = cfg.get('optimize_geo', True)
        self.output_path = cfg.get('output_path', 'results')
        
        # Store smooth barrier parameters from config
        smooth_barrier_param = cfg.get('smooth_barrier_param', {})
        self.smooth_eng_coeff = smooth_barrier_param.get('smooth_eng_coeff', 2e-4)
        self.barrier_coeff = smooth_barrier_param.get('barrier_coeff', 2e-4)
        self.increase_order_iter = smooth_barrier_param.get('increase_order_iter', 1000)
        
        # Create cache directory
        os.makedirs(self.cfg.get('tetwild_cache_folder', '.tetwild_cache'), exist_ok=True)
        
        # Load key points (sphere centers and radii)
        key_points_path = cfg.get('key_points_file_path', '')
        if key_points_path and os.path.exists(key_points_path):
            self._load_key_points(key_points_path)
            self._initialize_from_spheres()
        else:
            # Fallback to template mesh
            template_path = cfg.get('template_surface_sphere_path', 'mesh_data/s.1.obj')
            self.tetmesh = load_tetmesh_from_surface_mesh(template_path)
            
            # Create PyTorch parameter for optimization  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tet_v = torch.from_numpy(self.tetmesh.vertices).float().to(device)
            tet_v.requires_grad_(True)
            self.tet_v = torch.nn.Parameter(tet_v)
            
            # Store other mesh data as non-parameter tensors
            self.tet_elements = torch.from_numpy(self.tetmesh.elements).long().to(device)
            self.rest_matrices = torch.from_numpy(self.tetmesh.rest_matrices).float().to(device)
            self.surface_vid = torch.from_numpy(self.tetmesh.surface_vertices).long().to(device)
            self.surface_fid = torch.from_numpy(self.tetmesh.surface_faces).long().to(device)
            
            self.all_spheres_vtx_idx = None
            self.all_spheres_elem_idx = None
    
    def _load_key_points(self, key_points_path):
        """Load sphere centers and radii from JSON file"""
        with open(key_points_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Check for different dict formats
            if 'spheres' in data:
                # Format: {"spheres": [{"center": [x,y,z], "radius": r}, ...]}
                spheres = data['spheres']
                self.sphere_centers = np.array([s['center'] for s in spheres])
                self.sphere_radii = np.array([s['radius'] for s in spheres])
                print(f"DEBUG: Using 'spheres' format, found {len(spheres)} spheres")
            elif 'pt' in data and 'r' in data:
                # Format: {"pt": [[x,y,z], ...], "r": [[r], ...]} (your format)
                centers = np.array(data['pt'])
                radii = np.array(data['r'])
                
                print(f"DEBUG: Using 'pt'+'r' format, centers shape: {centers.shape}, radii shape: {radii.shape}")
                
                # Handle different radii formats
                if radii.ndim == 2 and radii.shape[1] == 1:
                    radii = radii.flatten()  # Convert [[r], [r], ...] to [r, r, ...]
                    print(f"DEBUG: Flattened radii shape: {radii.shape}")
                
                self.sphere_centers = centers
                self.sphere_radii = radii
            else:
                raise ValueError(f"Unsupported dict format in {key_points_path}. Expected 'spheres' or 'pt'+'r' keys.")
        elif isinstance(data, list):
            # Format: [[x,y,z,r], [x,y,z,r], ...]
            data = np.array(data)
            self.sphere_centers = data[:, :3]
            self.sphere_radii = data[:, 3]
            print(f"DEBUG: Using list format, data shape: {data.shape}")
        else:
            raise ValueError(f"Unsupported key points format in {key_points_path}")
            
        print(f"Loaded {len(self.sphere_centers)} spheres from {key_points_path}")
    
    def _initialize_from_spheres(self):
        """
        Initialize tetrahedral mesh from multiple spheres
        Replicates the original multi-sphere generation logic
        """
        template_sphere_path = self.cfg.get('template_surface_sphere_path', 'mesh_data/s.1.obj')
        
        if not os.path.exists(template_sphere_path):
            raise FileNotFoundError(f"Template sphere not found: {template_sphere_path}")
        
        template_sphere = trimesh.load(template_sphere_path)
        
        # Compute edge length for remeshing (from original logic)
        min_radius = min(self.sphere_radii)
        min_n_triangles = 100
        min_surface_area = min_radius * min_radius * math.pi
        min_triangle_area = min_surface_area / min_n_triangles
        edge_length_wrt_triangle_count = math.sqrt(min_triangle_area * 4.0 / math.sqrt(3))
        edge_length_wrt_bb = 0.03
        edge_length_min = 0.015
        final_edge_length = max(edge_length_min, min(edge_length_wrt_bb, edge_length_wrt_triangle_count))
        
        print(f"Using edge length: {final_edge_length}")
        
        # Generate tetrahedral mesh for each sphere
        all_vertices = []
        all_elements = []
        all_spheres_vtx_idx = []
        all_spheres_elem_idx = []
        
        base_vid = 0
        
        for sp_i, (center, radius) in enumerate(zip(self.sphere_centers, self.sphere_radii)):
            print(f"Processing sphere {sp_i}...")
            
            # Scale and translate template sphere
            sphere_vertices = template_sphere.vertices * radius + center
            
            # Generate tetrahedral mesh for this sphere
            tet = tetgen.TetGen(sphere_vertices, template_sphere.faces)
            tet.tetrahedralize(
                order=1,
                mindihedral=20,
                minratio=1.5,
                maxvolume=final_edge_length**3 / 6.0  # Control tet size
            )
            
            # Store vertices and elements
            sphere_vtx = tet.node.astype(np.float32)
            sphere_elem = tet.elem.astype(np.int32)
            
            all_vertices.append(sphere_vtx)
            all_elements.append(sphere_elem + base_vid)  # Offset element indices
            
            # Track sphere boundaries
            vtx_range = list(range(base_vid, base_vid + len(sphere_vtx)))
            all_spheres_vtx_idx.append(vtx_range)
            all_spheres_elem_idx.append(sphere_elem.tolist())
            
            base_vid += len(sphere_vtx)
            
            print(f"  Generated {len(sphere_vtx)} vertices, {len(sphere_elem)} tetrahedra")
        
        # Combine all spheres into single mesh
        combined_vertices = np.concatenate(all_vertices, axis=0)
        combined_elements = np.concatenate(all_elements, axis=0)
        
        print(f"Combined mesh: {len(combined_vertices)} vertices, {len(combined_elements)} tetrahedra")
        
        # Create WarpTetMesh
        self.tetmesh = WarpTetMesh(combined_vertices, combined_elements)
        
        # Create PyTorch parameter for optimization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tet_v = torch.from_numpy(combined_vertices).float().to(device)
        tet_v.requires_grad_(True)
        self.tet_v = torch.nn.Parameter(tet_v)
        
        # Store other mesh data as non-parameter tensors
        self.tet_elements = torch.from_numpy(combined_elements).long().to(device)
        self.rest_matrices = torch.from_numpy(self.tetmesh.rest_matrices).float().to(device)
        self.surface_vid = torch.from_numpy(self.tetmesh.surface_vertices).long().to(device)
        self.surface_fid = torch.from_numpy(self.tetmesh.surface_faces).long().to(device)
        
        # Store sphere tracking info for export
        self.all_spheres_vtx_idx = all_spheres_vtx_idx
        self.all_spheres_elem_idx = all_spheres_elem_idx
        
        # Save sphere info for later use
        os.makedirs(f"{self.output_path}/final", exist_ok=True)
        with open(f"{self.output_path}/final/spheres_vtx_idx.json", "w") as f:
            json.dump(all_spheres_vtx_idx, f, indent=4)
        with open(f"{self.output_path}/final/spheres_elem_idx.json", "w") as f:
            json.dump(all_spheres_elem_idx, f, indent=4)
    
    def forward(self, iter_num=0, **kwargs):
        """Forward pass - compatible with original interface and computes barrier energy"""
        if "permute_surface_v" in kwargs:
            dev = kwargs.get("permute_surface_v_dev", 0.01)
            print(f"Permute surface vertices with deviation {dev}")
            # Apply random perturbation to surface vertices
            # This would need to be implemented in WarpTetMesh
            pass
        
        # Compute barrier energy like in __call__ method
        from warp_tetmesh import WarpBarrierEnergy
        
        barrier_order = 3 if iter_num > self.increase_order_iter else 2
        
        barrier_energy = WarpBarrierEnergy.apply(
            self.tet_v,
            self.tet_elements.flatten().int(),
            self.rest_matrices,
            self.smooth_eng_coeff,  # Use config value
            self.barrier_coeff,     # Use config value
            barrier_order           # Increase order after specified iteration
        )
        
        # Use the optimizable parameter vertices
        tet_v = self.tet_v
        tet_elem = self.tet_elements
        surface_vid = self.surface_vid
        surface_f = self.surface_fid
        
        # Return forward data compatible with original interface
        class ForwardData:
            def __init__(self, tet_v, tet_elem, surface_vid, surface_f, barrier_energy):
                self.tet_v = tet_v
                self.tet_elem = tet_elem
                self.v_pos = tet_v[surface_vid] if surface_vid is not None else tet_v
                self.t_pos_idx = surface_f
                # Use computed barrier energy
                self.smooth_barrier_energy = barrier_energy
                
        return ForwardData(tet_v, tet_elem, surface_vid, surface_f, barrier_energy)
    
    def parameters(self):
        """Return trainable parameters"""
        import torch
        # Convert to torch tensor with gradient
        vertices_tensor = torch.from_numpy(self.tetmesh.vertices).float()
        vertices_tensor.requires_grad_(True)
        return [vertices_tensor]
    
    def export(self, path, name, save_npy=False):
        """Export mesh and sphere-specific data"""
        os.makedirs(path, exist_ok=True)
        
        # Export main mesh
        self.tetmesh.export_mesh(os.path.join(path, f"{name}.obj"))
        
        if save_npy:
            np.save(os.path.join(path, f"{name}_vertices.npy"), self.tetmesh.vertices)
            np.save(os.path.join(path, f"{name}_elements.npy"), self.tetmesh.elements)
        
        # Export individual sphere data (matching original behavior)
        if self.all_spheres_vtx_idx is not None:
            for i, (vtx_idx, elem_idx) in enumerate(zip(self.all_spheres_vtx_idx, self.all_spheres_elem_idx)):
                sphere_vertices = self.tetmesh.vertices[vtx_idx]
                sphere_elements = np.array(elem_idx)
                
                if save_npy:
                    np.save(os.path.join(path, f"{name}_sp{i}_vtx.npy"), sphere_vertices)
                    np.save(os.path.join(path, f"{name}_sp{i}_elem.npy"), sphere_elements)


class WarpTetMeshFish:
    """
    Warp implementation of TetMeshFish
    Placeholder for fish-specific geometry
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        # Implement fish-specific initialization logic here
        # For now, fallback to basic tetmesh loading
        template_path = cfg.get('template_surface_sphere_path', 'mesh_data/s.1.obj')
        self.tetmesh = load_tetmesh_from_surface_mesh(template_path)
    
    def forward(self, iter_num, **kwargs):
        """Forward pass - compatible with original interface"""
        if "permute_surface_v" in kwargs:
            dev = kwargs.get("permute_surface_v_dev", 0.01)
            print(f"Fish geometry: Permute surface vertices with deviation {dev}")
            # Apply random perturbation to surface vertices
            pass
        
        # Return forward data compatible with original interface  
        class ForwardData:
            def __init__(self, tet_v, tet_elem, surface_vid, surface_f):
                self.tet_v = tet_v
                self.tet_elem = tet_elem
                self.v_pos = tet_v[surface_vid] if surface_vid is not None else tet_v
                self.t_pos_idx = surface_f
                # Initialize smooth_barrier_energy properly - None will be handled by rasterizer
                self.smooth_barrier_energy = None
                
        # Use current tetmesh data (placeholder - needs proper implementation)
        import torch
        tet_v = torch.from_numpy(self.tetmesh.vertices).float()
        tet_elem = torch.from_numpy(self.tetmesh.elements).long()
        surface_vid = torch.arange(tet_v.shape[0])  # Placeholder
        surface_f = torch.zeros(0, 3, dtype=torch.long)  # Placeholder
        
        return ForwardData(tet_v, tet_elem, surface_vid, surface_f)
    
    def parameters(self):
        import torch
        vertices_tensor = torch.from_numpy(self.tetmesh.vertices).float()
        vertices_tensor.requires_grad_(True)
        return [vertices_tensor]
    
    def export(self, path, name, save_npy=False):
        os.makedirs(path, exist_ok=True)
        self.tetmesh.export_mesh(os.path.join(path, f"{name}.obj"))


def load_warp_geometry(geometry_class_type):
    """
    Dynamic geometry loading - fully compatible with original load_geometry
    """
    if geometry_class_type == "TetMeshFish":
        return WarpTetMeshFish
    elif geometry_class_type == "TetMeshMultiSphereGeometry":
        return WarpTetMeshMultiSphereGeometry
    else:
        raise NotImplementedError(
            f"Unknown geometry class type: {geometry_class_type}")


# Backward compatibility
WarpTetMeshGeometry = WarpTetMeshMultiSphereGeometry