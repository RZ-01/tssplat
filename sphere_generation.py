"""
High-quality sphere generation for TetSphere mesh fitting.

This module provides sphere generation functionality similar to the original 
generate_init_spheres.py but adapted for direct mesh input without requiring 
multi-view images.
"""

import os
import numpy as np
import torch
import trimesh
from scipy.optimize import milp, Bounds, LinearConstraint

# Import skeleton generation functions from data.utils
from data.utils import get_min_sdf_skel


def generate_init_spheres_from_mesh(mesh_path, surf_res=200, pc_res=200, radius_scale=1.1, offset=0.06, remesh_edge_length=0.08):
    import pypgo
    
    # Load target mesh
    mesh = trimesh.load(mesh_path, force='mesh')  # Force merging into single mesh
    
    # Handle both Mesh and Scene objects  
    if isinstance(mesh, trimesh.Scene):
        print(f"Scene contains {len(mesh.geometry)} geometries, merging...")
        meshes = list(mesh.geometry.values())
        if len(meshes) > 1:
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = meshes[0]
    
    # Handle both Mesh and Scene objects
    if hasattr(mesh, 'vertices'):
        vertices = mesh.vertices
        faces = mesh.faces
    else:
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in mesh file: {mesh_path}")
        first_mesh = list(mesh.geometry.values())[0]
        vertices = first_mesh.vertices
        faces = first_mesh.faces
    
    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create mesh object
    input_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Normalize mesh to [0, 1] range
    min_v = vertices.min(axis=0, keepdims=True)
    max_v = vertices.max(axis=0, keepdims=True)
    scale = (max_v - min_v).max()
    vertices_normalized = (vertices - min_v) / scale
    input_mesh = trimesh.Trimesh(vertices=vertices_normalized, faces=faces)
    
    print(f"Using surf_res={surf_res}, pc_res={pc_res} for sphere generation")
    trimeshgeo = pypgo.create_trimeshgeo(
            input_mesh.vertices.flatten(), input_mesh.faces.flatten())

    trimesh_remeshed = pypgo.mesh_isotropic_remeshing(
            trimeshgeo, remesh_edge_length, 5, 180)
    tri_v = pypgo.trimeshgeo_get_vertices(trimesh_remeshed)
    tri_t = pypgo.trimeshgeo_get_triangles(trimesh_remeshed)
    input_mesh_remeshed = trimesh.Trimesh(vertices=tri_v.reshape(-1, 3), faces=tri_t.reshape(-1, 3))
        
    print(f"Remeshed mesh: {len(input_mesh_remeshed.vertices)} vertices, {len(input_mesh_remeshed.faces)} faces")
        
        # Step 2: Generate skeleton points using original code's API
    print("Computing skeleton using original code's API...")
    skeleton = get_min_sdf_skel(input_mesh_remeshed)
    candidate_points = skeleton
        
    if isinstance(candidate_points, torch.Tensor):
        candidate_points = candidate_points.cpu().numpy()
        
    print(f"Generated {len(candidate_points)} skeleton points")
        
        # Step 3: Use MILP optimization to select optimal spheres (like original code)
    print("Optimizing sphere selection using MILP...")
        
        # Convert to torch tensors for distance computation
    point_set = torch.tensor(input_mesh.vertices).cuda().double()
    inner_set = torch.tensor(candidate_points).cuda().double()
        
        # Compute distances and radii
    dist = batch_cdist(inner_set, point_set, batch_size=500)
    radius = dist.topk(10, largest=False).values.mean(dim=1, keepdim=True)
    radius_scaled = radius * radius_scale + offset
        
        # Solve MILP optimization
    options = {"disp": True, "time_limit": 30000, "mip_rel_gap": 0.20}
    res_milp, D, point_set_filtered = solve_milp(inner_set, point_set, radius, radius_scale, offset, options)
        
    res_milp.x = [int(x_i) for x_i in res_milp.x]
    binary_x = res_milp.x
    value_pos = np.nonzero(binary_x)[0]
    print("Phase 1 selected spheres: ", np.sum(binary_x))
        
        # Check for uncovered points and solve again if needed
    covered_flag = D @ binary_x
    uncovered_flag = covered_flag < 0.5
    uncovered_points = point_set_filtered[uncovered_flag]
    print(f"Uncovered points: {np.sum(uncovered_flag)}")
        
    if np.sum(uncovered_flag) > 0:
        print("Solving phase 2 for uncovered points...")
        options = {"disp": True, "time_limit": 30000, "mip_rel_gap": 0.0}
        res_milp_2, _, _ = solve_milp(inner_set, uncovered_points, radius, radius_scale, offset, options)
        res_milp_2.x = [int(x_i) for x_i in res_milp_2.x]
        binary_x_2 = res_milp_2.x
        print("Phase 2 selected spheres: ", np.sum(binary_x_2))
        value_pos_2 = np.nonzero(binary_x_2)[0]
            
        value_pos = np.concatenate([value_pos, value_pos_2])
        
    print(f"Total selected spheres: {len(value_pos)}")
        
        # Get final sphere centers and radii
    final_selected_pts = inner_set.cpu().numpy()[value_pos]
    radius_scaled_final = radius_scaled.cpu().numpy()[value_pos]
        # Add small additional offset for better coverage
    radius_scaled_final += offset * 0.3
        
    print(f"Generated {len(final_selected_pts)} spheres with radii range: [{radius_scaled_final.min():.4f}, {radius_scaled_final.max():.4f}]")
        
    return {
            "pt": final_selected_pts.tolist(),
            "r": radius_scaled_final.tolist()
        }

    

        




def batch_cdist(x, y, batch_size):
    """Compute pairwise distances in batches to avoid OOM."""
    return torch.cat(
        [
            torch.cdist(x[start: start + batch_size], y)
            for start in range(0, x.shape[0], batch_size)
        ],
        dim=0,
    )


def batch_construct_D(radius, dist, batch_size):
    """Construct coverage matrix D in batches."""
    return torch.cat(
        [
            torch.gt(radius, dist[start: start + batch_size]).type(torch.int)
            for start in range(0, dist.shape[0], batch_size)
        ],
        dim=0,
    )


def solve_milp(inner_set, point_set, radius, radius_scale, offset, options):
    """Solve MILP optimization for sphere selection."""
    dist = batch_cdist(inner_set, point_set, batch_size=500)
    radius_scaled = radius * radius_scale + offset

    dist = dist.permute(1, 0)  # [N, Nin]
    radius_g = radius_scaled.permute(1, 0)  # [1, Nin]

    D = batch_construct_D(radius_g, dist, batch_size=500)
    D = D.cpu().numpy()
    
    # Filter out zero rows (points that can't be covered)
    zero_rows = np.all(D == 0, axis=1)
    zero_row_count = np.sum(zero_rows)
    print(f"Zero row count: {zero_row_count}")
    if zero_row_count < 200:  # Only filter if reasonable number of zero rows
        D = D[~zero_rows]
        point_set = point_set[~zero_rows]

    c = np.ones(len(inner_set))
    A, b = D, np.ones(len(point_set))
    integrality = np.ones(len(inner_set))
    lb, ub = np.zeros(len(inner_set)), np.ones(len(inner_set))
    variable_bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, lb=b)
    res_milp = milp(
        c,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options)
    
    return res_milp, D, point_set


def load_target_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')  # Force merging into single mesh

    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        if len(meshes) > 1:
            print(f"Merging {len(meshes)} geometries into single mesh...")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = meshes[0]

    vertices = np.array(mesh.vertices, dtype=np.float32)
    
    # Normalize to [0, 1] range
    min_v = vertices.min(axis=0, keepdims=True)
    max_v = vertices.max(axis=0, keepdims=True)
    scale = (max_v - min_v).max()
    norm_vertices = (vertices - min_v) / scale

    return torch.from_numpy(norm_vertices)
