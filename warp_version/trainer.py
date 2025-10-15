import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import trange, tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from materials import load_material
    MATERIALS_AVAILABLE = True
except ImportError:
    MATERIALS_AVAILABLE = False
    print("Warning: Materials not available (nvdiffrast not installed)")

from utils.config import load_config
from utils.optimizer import AdamUniform
from warp_mesh_rasterizer import WarpMeshRasterizer
from warp_image_loss import WarpImageLoss
from warp_tetmesh import WarpTetMesh, load_tetmesh_from_surface_mesh
from warp_geometry import load_warp_geometry
from warp_dataloader import load_warp_dataloader


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


def train_warp_tetsplat(cfg):
    verbose = cfg.get("verbose", False)
    material = None
    os.makedirs(os.path.join(cfg.output_path, "final/"), exist_ok=True)
    
    shade_loss = torch.nn.MSELoss()
    if cfg.get("fitting_stage", None) == "texture":
        if not MATERIALS_AVAILABLE:
            raise ImportError("Materials not available for texture fitting stage. Please install nvdiffrast.")
        assert cfg.get("material", None) is not None
        material = load_material(cfg.material_type)(cfg.material)
        shade_loss = torch.nn.L1Loss()
    
    geometry_class = load_warp_geometry(cfg.geometry_type)
    geometry = geometry_class(cfg.geometry)
    
    renderer = WarpMeshRasterizer(geometry, material, cfg.renderer)
    
    dataloader_mapping = {
        "MistubaImgDataLoader": "mitsuba",
        "Wonder3DDataLoader": "wonder3d", 
        "BlenderDataLoader": "blender",
        "GSO_DataLoader": "gso",
        "NeRFDataLoader": "nerf"
    }
    
    warp_dataloader_type = dataloader_mapping.get(cfg.dataloader_type, cfg.dataloader_type)
    
    try:
        dataset_class = load_warp_dataloader(warp_dataloader_type)
    except (NotImplementedError, ImportError) as e:
        print(f"Warning: Warp dataloader '{warp_dataloader_type}' not found, trying lowercase...")
        try:
            dataset_class = load_warp_dataloader(cfg.dataloader_type.lower())
        except:
            print(f"Error: No Warp implementation found for dataloader '{cfg.dataloader_type}'")
            print("Available Warp dataloaders:", list(dataloader_mapping.values()))
            raise NotImplementedError(f"Unsupported dataloader type: {cfg.dataloader_type}")
    
    dataloader = dataset_class(cfg.data)
    num_forward_per_iter = dataloader.num_forward_per_iter
    
    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.total_num_iter * num_forward_per_iter, eta_min=1e-4)
    
    permute_surface_scheduler = None
    if cfg.get('use_permute_surface_v', False):
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    best_loss = 1e10
    best_loss_iter = 0
    best_opt_imgs = None
    best_v = None
    
    print("Starting Warp TetSplatting training...")
    print(f"Total iterations: {cfg.total_num_iter}")
    print(f"Forward passes per iteration: {num_forward_per_iter}")
    print(f"Dataset size: {len(dataloader)}")
    
    for it in trange(cfg.total_num_iter):
        for forw_id in range(num_forward_per_iter):
            batch = dataloader(it, forw_id)
            
            color_ref = batch["img"]
            
            fit_depth = cfg.get("fit_depth", False)
            if fit_depth:
                fit_depth = cfg.get("fit_depth_starting_iter", 0) < it
            
            renderer_input = {
                "mvp": batch["mvp"],
                "only_alpha": cfg.get("fitting_stage", None) == "geometry",
                "iter_num": it,
                "resolution": batch["resolution"],
                "background": batch["background"],
                "permute_surface_scheduler": permute_surface_scheduler,
                "fit_depth": fit_depth,
                "campos": batch["campos"],
            }
            
            out = renderer(**renderer_input)
            
            if cfg.get("fitting_stage", None) == "geometry":
                img_loss = shade_loss(out["shaded"][..., -1], color_ref[..., -1])
            else:
                img_loss = shade_loss(out["shaded"][..., :3], color_ref[..., :3])
            
            img_loss *= 20
            
            if fit_depth:
                img_loss += shade_loss(out["d"][..., -1] * color_ref[..., -1],
                                       batch["d"][..., -1] * color_ref[..., -1]) * 100
            
            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out.get("geo_regularization", 0.0)
                if reg_loss == 0.0 and hasattr(out, "smooth_barrier_energy"):
                    reg_loss = out.smooth_barrier_energy
            
            loss = img_loss * 100 + reg_loss
            
            if True:
                tqdm.write(
                    "iter=%4d, img_loss=%.4f, reg_loss=%.4f"
                    % (it, img_loss, reg_loss)
                )
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            cur_loss = loss.clone().detach().cpu().item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_loss_iter = it
                if hasattr(geometry, 'tet_v'):
                    best_v = geometry.tet_v.clone().detach()
                best_opt_imgs = out["shaded"].clone().detach()
            
            if it % 100 == 0 and forw_id == 0:
                os.makedirs(f"{cfg.output_path}/mesh{it:05d}", exist_ok=True)
                geometry.export(f"{cfg.output_path}/mesh{it:05d}", f"{it:05d}")
                
                if verbose:
                    chosen_idx = np.random.randint(0, batch["img"].shape[0])
                    opt_img = out["shaded"][chosen_idx].clone().detach()
                    img = opt_img.cpu().numpy()

                    print(img.shape)
                    if img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)

                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_ours-{}.png".format(cfg.output_path, it))

                    img = color_ref[chosen_idx].cpu().numpy()
                    print(img.shape)
                    if img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)

                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_gt-{}.png".format(cfg.output_path, it))

                    if cfg.get("fitting_stage", None) == "geometry":
                        diff = color_ref[chosen_idx].cpu().numpy()[..., -1:] - opt_img.cpu().numpy()[..., -1:]
                        img = np.abs(diff)
                        print(img.shape)
                        if img.shape[2] == 1:
                            img = np.concatenate([img, img, img, img], axis=2)

                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save("{}/a_diff-{}.png".format(cfg.output_path, it))
    
    print(f"Best rendering loss: {best_loss} at iteration {best_loss_iter}")
    
    geometry.export(f"{cfg.output_path}/final", "final", save_npy=True)
    
    if material is not None:
        material.export(f"{cfg.output_path}/final", "material")
        if hasattr(renderer, 'export'):
            renderer.export(f"{cfg.output_path}/final", "material")
    
    return {
        "best_loss": best_loss,
        "best_loss_iter": best_loss_iter,
        "final_geometry": geometry
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp-based TetSplatting trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    
    args, extras = parser.parse_known_args()
    
    cfg = load_config(args.config, cli_args=extras)
    print("Starting Warp TetSplatting training with official config...")
    print(f"Config: {args.config}")
    print(f"Experiment: {cfg.expr_name}")
    print(f"Geometry type: {cfg.geometry_type}")
    print(f"Dataloader type: {cfg.dataloader_type}")
    
    results = train_warp_tetsplat(cfg)
    print("Training completed!")
    print(f"Best loss: {results['best_loss']} at iteration {results['best_loss_iter']}")