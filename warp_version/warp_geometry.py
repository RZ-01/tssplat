import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from warp_tetmesh import load_tetmesh_from_surface_mesh
import numpy as np
import json
import trimesh
import torch


class WarpTetMeshGeometry(torch.nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.optimize_geo = cfg.get('optimize_geo', True)
        self.output_path = cfg.get('output_path', 'results')
        
        smooth_barrier_param = cfg.get('smooth_barrier_param', {})
        self.smooth_eng_coeff = smooth_barrier_param.get('smooth_eng_coeff', 2e-4)
        self.barrier_coeff = smooth_barrier_param.get('barrier_coeff', 2e-4)
        self.increase_order_iter = smooth_barrier_param.get('increase_order_iter', 1000)
        
        template_path = cfg.get('template_surface_sphere_path', 'mesh_data/s.1.obj')
        if os.path.exists(template_path):
            self.tetmesh = load_tetmesh_from_surface_mesh(template_path)
        else:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            self.tetmesh = load_tetmesh_from_surface_mesh(sphere)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tet_v = torch.from_numpy(self.tetmesh.vertices).float().to(device)
        if self.optimize_geo:
            self.tet_v = torch.nn.Parameter(tet_v, requires_grad=True)
        else:
            self.register_buffer("tet_v", tet_v)
        
        self.tet_elements = torch.from_numpy(self.tetmesh.elements).long().to(device)
        self.surface_vid = torch.from_numpy(self.tetmesh.surface_vertices).long().to(device)
        self.surface_fid = torch.from_numpy(self.tetmesh.surface_faces).long().to(device)
        
        print(f"WarpTetMeshGeometry initialized: {len(self.tetmesh.vertices)} vertices, {len(self.tetmesh.elements)} tetrahedra")
    
    def coeff_scheduler(self, iter_num):
        import math
        
        smooth_coeff = self.smooth_eng_coeff
        barrier_coeff = self.barrier_coeff
        multiplier = math.pow(
            2,
            abs(math.sin(min(iter_num / 300.0 / 4 * 0.5 * math.pi, 0.5 * math.pi)))
            * 4,
        )
        
        smooth_coeff *= multiplier
        barrier_coeff *= multiplier
        return smooth_coeff, barrier_coeff

    def forward(self, iter_num=0, **kwargs):
        if "permute_surface_v" in kwargs:
            dev = kwargs.get("permute_surface_v_dev", 0.01)
            print(f"Permute surface vertices with deviation {dev}")
            pass
        
        smooth_coeff, barrier_coeff = self.coeff_scheduler(iter_num)
        
        from tet_spheres import tet_spheres_ext
        import numpy as np
        
        if not hasattr(self, 'tet_sp'):
            v_flat = self.tet_v.detach().cpu().numpy().flatten().astype(np.float32)
            f_flat = self.tet_elements.cpu().numpy().flatten().astype(np.int32)
            self.tet_sp = tet_spheres_ext.TetSpheres(v_flat, f_flat)
        
        barrier_order = 4 if iter_num > self.increase_order_iter else 2
        
        total_reg_energy = tet_spheres_ext.forward(
            self.tet_v,
            self.tet_sp,
            smooth_coeff,
            barrier_coeff,
            barrier_order
        )
        
        if iter_num % 100 == 0:
            print(f"[CUDA REG DEBUG iter={iter_num}] "
                  f"smooth_coeff={smooth_coeff:.6f}, barrier_coeff={barrier_coeff:.6f}, "
                  f"order={barrier_order}, total_reg={total_reg_energy.item():.6f}")
        
        tet_v = self.tet_v
        tet_elem = self.tet_elements
        surface_vid = self.surface_vid
        surface_f = self.surface_fid
        
        class ForwardData:
            def __init__(self, tet_v, tet_elem, surface_vid, surface_f, total_energy):
                self.tet_v = tet_v
                self.tet_elem = tet_elem
                
                self.v_pos = tet_v[surface_vid]
                self.t_pos_idx = surface_f
                
                self.smooth_barrier_energy = total_energy
                
        return ForwardData(tet_v, tet_elem, surface_vid, surface_f, total_reg_energy)
    
    def export(self, path: str, filename: str, **kwargs):
        os.makedirs(path, exist_ok=True)
        
        surface_mesh = self.tetmesh.get_surface_mesh()
        
        current_vertices = self.tet_v.detach().cpu().numpy()
        surface_vertices = current_vertices[self.tetmesh.surface_vertices]
        surface_mesh.vertices = surface_vertices
        
        obj_path = os.path.join(path, f"{filename}.obj")
        surface_mesh.export(obj_path)
        print(f"Exported surface mesh to {obj_path}")


class WarpTetMeshMultiSphereGeometry(WarpTetMeshGeometry):

    
    def __init__(self, cfg):
        super().__init__(cfg)

        template_path = cfg.get('template_surface_sphere_path', 'mesh_data/s.1.obj')
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template sphere not found: {template_path}")
        
        template_mesh = trimesh.load(template_path)
        
        init_data_path = cfg.get('sphere_init_data_path') or cfg.get('key_points_file_path')
        if init_data_path and os.path.exists(init_data_path):
            with open(init_data_path, 'r') as f:
                init_data = json.load(f)
            
            combined_vertices = []
            combined_faces = []
            vertex_offset = 0
            
            for sphere_data in init_data.get('spheres', []):
                center = np.array(sphere_data.get('center', [0, 0, 0]))
                radius = sphere_data.get('radius', 1.0)
                
                sphere_vertices = template_mesh.vertices * radius + center
                sphere_faces = template_mesh.faces + vertex_offset
                
                combined_vertices.append(sphere_vertices)
                combined_faces.append(sphere_faces)
                vertex_offset += len(sphere_vertices)
            
            if combined_vertices:
                all_vertices = np.vstack(combined_vertices)
                all_faces = np.vstack(combined_faces)
                combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
            else:
                combined_mesh = template_mesh
        else:
            combined_mesh = template_mesh
        
        self.tetmesh = load_tetmesh_from_surface_mesh(combined_mesh)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tet_v = torch.from_numpy(self.tetmesh.vertices).float().to(device)
        if self.optimize_geo:
            self.tet_v = torch.nn.Parameter(tet_v, requires_grad=True)
        else:
            self.register_buffer("tet_v", tet_v)
        
        self.tet_elements = torch.from_numpy(self.tetmesh.elements).long().to(device)
        self.surface_vid = torch.from_numpy(self.tetmesh.surface_vertices).long().to(device)
        self.surface_fid = torch.from_numpy(self.tetmesh.surface_faces).long().to(device)
        
        print(f"WarpTetMeshMultiSphereGeometry initialized: {len(self.tetmesh.vertices)} vertices, {len(self.tetmesh.elements)} tetrahedra")
    
    def coeff_scheduler(self, iter_num):
        import math
        
        smooth_coeff = self.smooth_eng_coeff
        barrier_coeff = self.barrier_coeff
        multiplier = math.pow(
            2,
            abs(math.sin(min(iter_num / 300.0 / 4 * 0.5 * math.pi, 0.5 * math.pi)))
            * 4,
        )
        
        smooth_coeff *= multiplier
        barrier_coeff *= multiplier
        return smooth_coeff, barrier_coeff

    def forward(self, iter_num=0, **kwargs):
        if "permute_surface_v" in kwargs:
            dev = kwargs.get("permute_surface_v_dev", 0.01)
            print(f"Permute surface vertices with deviation {dev}")
            pass
        
        smooth_coeff, barrier_coeff = self.coeff_scheduler(iter_num)
        
        from tet_spheres import tet_spheres_ext
        import numpy as np
        
        if not hasattr(self, 'tet_sp'):
            v_flat = self.tet_v.detach().cpu().numpy().flatten().astype(np.float32)
            f_flat = self.tet_elements.cpu().numpy().flatten().astype(np.int32)
            self.tet_sp = tet_spheres_ext.TetSpheres(v_flat, f_flat)
        
        barrier_order = 4 if iter_num > self.increase_order_iter else 2
        
        total_reg_energy = tet_spheres_ext.forward(
            self.tet_v,
            self.tet_sp,
            smooth_coeff,
            barrier_coeff,
            barrier_order
        )
        
        
        tet_v = self.tet_v
        tet_elem = self.tet_elements
        surface_vid = self.surface_vid
        surface_f = self.surface_fid
        
        class ForwardData:
            def __init__(self, tet_v, tet_elem, surface_vid, surface_f, total_energy):
                self.tet_v = tet_v
                self.tet_elem = tet_elem
                
                self.v_pos = tet_v[surface_vid]
                self.t_pos_idx = surface_f
                
                self.smooth_barrier_energy = total_energy
                
        return ForwardData(tet_v, tet_elem, surface_vid, surface_f, total_reg_energy)


def load_warp_geometry(geometry_class_type):
    return load_geometry(geometry_class_type)

def load_geometry(geometry_class_type):
    if geometry_class_type in ["WarpTetMeshGeometry", "TetMeshGeometry"]:
        return WarpTetMeshGeometry
    elif geometry_class_type in ["WarpTetMeshMultiSphereGeometry", "TetMeshMultiSphereGeometry"]:
        return WarpTetMeshMultiSphereGeometry
    else:
        raise ValueError(f"Unknown geometry class type: {geometry_class_type}. "
                        f"Supported types: TetMeshGeometry, TetMeshMultiSphereGeometry, "
                        f"WarpTetMeshGeometry, WarpTetMeshMultiSphereGeometry")
