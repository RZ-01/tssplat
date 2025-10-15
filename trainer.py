import os
from tqdm import trange
import argparse
import torch
import json
import numpy as np

from geometry import load_geometry
from materials import load_material
from renderers import MeshRasterizer
from data import load_dataloader
from utils.config import load_config
from utils.optimizer import AdamUniform
from pytorch3d.ops import knn_points
from sphere_generation import generate_init_spheres_from_mesh, load_target_mesh

def visualize_correspondences(
    pred_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    output_path: str,
    iteration: int,
    k: int = 1,
    max_pairs: int = 5000,
    sample_ratio: float = 0.1,
):
    
    pred_v = pred_vertices.detach().cpu()
    target_v = target_vertices.detach().cpu()
    
    N = pred_v.shape[0]
    n_sample = min(max_pairs, int(N * sample_ratio))
    if n_sample < N:
        indices = torch.randperm(N)[:n_sample]
        pred_v_sampled = pred_v[indices]
    else:
        pred_v_sampled = pred_v
    
    pred_v_batch = pred_v_sampled.unsqueeze(0)  # [1, n, 3]
    target_v_batch = target_v.unsqueeze(0)  # [1, M, 3]
    
    _, idx, nn = knn_points(pred_v_batch, target_v_batch, K=k, return_nn=True)
    
    # 获取对应的target点
    nearest_targets = nn[0, :, 0, :]  # [n, 3] 取k=1的最近邻
    
    # 转换为numpy
    pred_np = pred_v_sampled.numpy()
    target_np = nearest_targets.numpy()
    
    # 创建输出目录
    viz_dir = os.path.join(output_path, "viz_correspondences")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 构建.obj文件
    obj_path = os.path.join(viz_dir, f"correspondence_iter_{iteration:04d}.obj")
    
    with open(obj_path, 'w') as f:
        f.write(f"# Correspondence visualization for iteration {iteration}\n")
        f.write(f"# Total pairs: {len(pred_np)}\n\n")
        
        vertex_idx = 1
        
        for i in range(len(pred_np)):
            p = pred_np[i]  # pred point
            t = target_np[i]  # target point
            
            # 计算垂直于连线的一个小偏移向量（用于构造三角形）
            vec = t - p
            length = np.linalg.norm(vec)
            
            if length < 1e-8:
                continue
            
            # 创建一个垂直向量（非常小，形成细长三角形）
            perp = np.array([-vec[1], vec[0], 0])
            if np.linalg.norm(perp) < 1e-8:
                perp = np.array([0, -vec[2], vec[1]])
            perp = perp / (np.linalg.norm(perp) + 1e-8) * (length * 0.001)  # 很细的偏移
            
            # 三角形的三个顶点：pred点 + target点 + 一个微小偏移的中间点
            mid = (p + t) / 2 + perp
            
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")  # pred点
            f.write(f"v {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")  # target点
            f.write(f"v {mid[0]:.6f} {mid[1]:.6f} {mid[2]:.6f}\n")  # 中间点
            
            # 创建三角形面
            f.write(f"f {vertex_idx} {vertex_idx+1} {vertex_idx+2}\n")
            vertex_idx += 3
    
    # 计算平均距离用于显示
    distances = np.linalg.norm(pred_np - target_np, axis=1)
    mean_dist = distances.mean()
    
    print(f"[Iter {iteration}] Saved correspondence viz: {obj_path} (mean_dist: {mean_dist:.6f})")
    
    return mean_dist

def compute_mesh_distance_loss(
    pred_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    k: int = 1,
    squared: bool = True,
    batch_size: int = 20000,
    sample_ratio: float = 1.0,
    debug: bool = False,
) -> torch.Tensor:
    """
    双向 Chamfer 距离：Pred <-> Target
    pred_vertices: [N, 3]  target_vertices: [M, 3]
    返回：标量 loss
    """
    device = pred_vertices.device
    assert target_vertices.device == device, "pred/target 必须在同一设备上"

    # 对预测点下采样（可选）
    if sample_ratio < 1.0:
        N = pred_vertices.shape[0]
        n_sample = max(100, int(N * sample_ratio))
        n_sample = min(n_sample, N)
        idx = torch.randperm(N, device=device)[:n_sample]
        pred_vertices = pred_vertices[idx]
        if debug:
            print(f"[chamfer_loss] sample {n_sample}/{N} ({sample_ratio*100:.1f}%)")

    Np = pred_vertices.shape[0]
    Nm = target_vertices.shape[0]
    k_use = max(1, int(k))

    # 分批分别算 Pred->Target 和 Target->Pred
    # Pred -> Target
    means_pred = []
    T = target_vertices.unsqueeze(0)  # [1, M, 3]
    for s in range(0, Np, batch_size):
        P = pred_vertices[s:s+batch_size].unsqueeze(0)  # [1, b, 3]
        d2, _, _ = knn_points(P, T, K=k_use, return_nn=False)
        d2 = d2.mean(dim=-1) if k > 1 else d2[..., 0]  # [1, b]
        d = d2 if squared else torch.sqrt(d2 + 1e-12)
        means_pred.append(d.mean())
    loss_pred = torch.stack(means_pred).mean()

    # Target -> Pred
    means_target = []
    P = pred_vertices.unsqueeze(0)  # [1, Np, 3]
    for s in range(0, Nm, batch_size):
        T_part = target_vertices[s:s+batch_size].unsqueeze(0)  # [1, b, 3]
        d2, _, _ = knn_points(T_part, P, K=k_use, return_nn=False)
        d2 = d2.mean(dim=-1) if k > 1 else d2[...,0]
        d = d2 if squared else torch.sqrt(d2 + 1e-12)
        means_target.append(d.mean())
    loss_target = torch.stack(means_target).mean()

    loss = 0.5 * (loss_pred + loss_target)

    if debug:
        print(
            f"[chamfer_loss] k={k}, squared={squared}, loss_pred={loss_pred.item():.6f}, "
            f"loss_target={loss_target.item():.6f}, chamfer={loss.item():.6f}"
        )

    return loss



class LinearInterpolateScheduler:
    def __init__(self, start_iter, end_iter, start_val, end_val, freq):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_val = start_val
        self.end_val = end_val
        self.freq = freq

    def __call__(self, iter):
        if iter < self.start_iter or iter % self.freq != 0 or iter == 0:
            return None

        p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_val * (1 - p) + self.end_val * p


def train(cfg):
    #verbose = cfg.get("verbose", False)
    
    # Load target mesh if provided
    target_vertices = None
    if cfg.get("target_mesh_path", None) is not None:
        target_vertices = load_target_mesh(cfg.target_mesh_path)
        print(f"Loaded target mesh from {cfg.target_mesh_path} with {target_vertices.shape[0]} vertices")
        
        # Generate initial spheres from target mesh
        mesh_data_dir = os.path.dirname(cfg.geometry.key_points_file_path)
        os.makedirs(mesh_data_dir, exist_ok=True)
        
        print("Generating initial spheres from target mesh...")
        surf_res = cfg.get("surf_res", 200)
        pc_res = cfg.get("pc_res", 200)
        radius_scale = cfg.get("radius_scale", 1.1)
        offset = cfg.get("offset", 0.06)
        remesh_edge_length = cfg.get("remesh_edge_length", 0.08)
        
        init_spheres = generate_init_spheres_from_mesh(
            cfg.target_mesh_path, 
            surf_res=surf_res,
            pc_res=pc_res,
            radius_scale=radius_scale,
            offset=offset,
            remesh_edge_length=remesh_edge_length
        )
        
        with open(cfg.geometry.key_points_file_path, 'w') as f:
            json.dump(init_spheres, f, indent=4)
        
        print(f"Generated {len(init_spheres['pt'])} initial spheres and saved to {cfg.geometry.key_points_file_path}")
    
    material = None
    cfg.geometry.optimize_geo = True
    cfg.geometry.output_path = cfg.output_path
    
    # Create output directory (required by geometry initialization)
    os.makedirs(os.path.join(cfg.output_path, "final"), exist_ok=True)

    # Only setup shade_loss for image fitting mode
    """
    shade_loss = None
    if target_vertices is None:  # Image fitting mode
        shade_loss = torch.nn.MSELoss()
        if cfg.get("fitting_stage", None) == "texture":
            assert cfg.get("material", None) is not None
            material = load_material(cfg.material_type)(cfg.material)
            cfg.geometry.optimize_geo = False
            shade_loss = torch.nn.L1Loss()
    """

    geometry = load_geometry(cfg.geometry_type)(cfg.geometry)
    renderer = MeshRasterizer(geometry, material, cfg.renderer)
    dataloader = load_dataloader(cfg.dataloader_type)(cfg.data)

    num_forward_per_iter = dataloader.num_forward_per_iter

    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.total_num_iter * num_forward_per_iter, eta_min=1e-4)

    permute_surface_scheduler = None
    if cfg.use_permute_surface_v:
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    # Move target vertices to device if available
    if target_vertices is not None:
        target_vertices = target_vertices.to(geometry.tet_v.device)

    # main loop
   # is_mesh_fitting = target_vertices is not None and cfg.get("fitting_stage", None) == "geometry"
    loss_batch_size = cfg.data.get("batch_size", 1000) 
    mesh_sample_ratio = cfg.get("mesh_loss_sample_ratio", 1.0)  


    pbar = trange(cfg.total_num_iter, desc="Training")
    
    for it in pbar:
        for forw_id in range(num_forward_per_iter):
            #batch = dataloader(it, forw_id)

            #fit_depth = cfg.get("fit_depth", False)
            #if fit_depth:
            #    fit_depth = cfg.get("fit_depth_starting_iter", 0) < it
            
            renderer_input = {
                "iter_num": it,
                "permute_surface_scheduler": permute_surface_scheduler,
            }

            # forward
            out = renderer.compute_geometry_forward(**renderer_input)
        
            # Compute loss
            current_surface_vertices = geometry.tet_v[geometry.surface_vid]
            raw_mesh_loss = compute_mesh_distance_loss(
                current_surface_vertices, 
                target_vertices, 
                batch_size=loss_batch_size,
                sample_ratio=mesh_sample_ratio
            )
            data_loss = raw_mesh_loss * cfg.get("mesh_loss_weight")

            """
            else:
                color_ref = batch["img"]
                if cfg.get("fitting_stage", None) == "geometry":
                    data_loss = shade_loss(out["shaded"][..., -1], color_ref[..., -1])
                else:
                    data_loss = shade_loss(out["shaded"][..., :3], color_ref[..., :3])
                data_loss *= 20
                
                if fit_depth:
                    data_loss += shade_loss(out["d"][..., -1] * color_ref[..., -1],
                                           batch["d"][..., -1] * color_ref[..., -1]) * 100
            """

            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out["geo_regularization"] * cfg.get("reg_loss_weight", 0.1)

            loss = data_loss + reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            optimizer.step()
            scheduler.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mesh': f'{raw_mesh_loss.item():.4f}',
                'reg': f'{reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:.4f}',
            })


    final_output_dir = os.path.join(cfg.output_path, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    geometry.export(final_output_dir, "final", save_npy=True)
    
    print(f"Final mesh saved to: {final_output_dir}")

    if material is not None:
        material.export(f"{cfg.output_path}/final", "material")
        renderer.export(f"{cfg.output_path}/final", "material")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    args, extras = parser.parse_known_args()
    
    cfg_file = args.config
    cfg = load_config(cfg_file, cli_args=extras)
    
    train(cfg)
