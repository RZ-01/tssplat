"""Configuration management for hybrid TetSplatting pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf


@dataclass
class HybridConfig:
    """Configuration for hybrid TetSplatting components"""
    use_warp_rasterization: bool = True
    use_original_geometry: bool = True
    use_original_materials: bool = True
    use_original_dataloader: bool = True
    fallback_to_original: bool = True
    enable_caching: bool = True
    cache_dir: str = ".hybrid_cache"
    verbose: bool = False
    debug_mode: bool = False


@dataclass
class HybridTrainingConfig:
    """Configuration for hybrid training"""
    total_num_iter: int = 10000
    batch_size: int = 1
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    geometry_type: str = "TetMeshMultiSphereGeometry"
    dataloader_type: str = "Wonder3DDataLoader"
    material_type: str = "explicit"
    output_path: str = "results/hybrid"
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 0.01,
        "grad_limit": True,
        "grad_limit_values": [0.01, 0.01],
        "grad_limit_iters": [1500]
    })


def create_hybrid_config(
    base_config_path: str,
    hybrid_overrides: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> DictConfig:
    """Create hybrid configuration by combining base config with overrides."""
    if base_config_path.endswith('.yaml') or base_config_path.endswith('.yml'):
        import yaml
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    else:
        raise ValueError("Base config must be a YAML file")
    
    hybrid_config = HybridTrainingConfig()
    
    if hybrid_overrides:
        for key, value in hybrid_overrides.items():
            if hasattr(hybrid_config, key):
                setattr(hybrid_config, key, value)
            elif key in hybrid_config.hybrid.__dict__:
                setattr(hybrid_config.hybrid, key, value)
    
    hybrid_dict = OmegaConf.structured(hybrid_config)
    hybrid_dict = OmegaConf.to_container(hybrid_dict, resolve=True)
    merged_config = {**base_config, **hybrid_dict}
    
    if output_path:
        merged_config['output_path'] = output_path
    
    return OmegaConf.create(merged_config)


def create_gso_hybrid_config(
    data_path: str,
    output_path: str = "results/gso_hybrid",
    use_warp_rasterization: bool = True,
    total_iterations: int = 10000
) -> DictConfig:
    """Create hybrid configuration for GSO dataset training."""
    hybrid_overrides = {
        "use_warp_rasterization": use_warp_rasterization,
        "use_original_geometry": True,
        "use_original_materials": True,
        "use_original_dataloader": True,
        "fallback_to_original": True,
        "total_num_iter": total_iterations,
        "output_path": output_path,
        "verbose": True
    }
    
    base_config = {
        "expr_name": "gso_hybrid",
        "fitting_stage": "geometry",
        
        "geometry_type": "TetMeshMultiSphereGeometry",
        "geometry": {
            "initial_mesh_path": "",
            "use_smooth_barrier": True,
            "smooth_barrier_param": {
                "smooth_eng_coeff": 2e-4,
                "barrier_coeff": 2e-4,
                "increase_order_iter": 1000
            },
            "template_surface_sphere_path": "mesh_data/s.1.obj",
            "key_points_file_path": "mesh_data/${expr_name}/${expr_name}.json",
            "tetwild_exec": "/path/to/TetWild/build/TetWild",
            "tetwild_cache_folder": ".tetwild_cache",
            "load_precomputed_tetwild_mesh": False,
            "debug_mode": False
        },
        
        "material_type": None,
        
        "dataloader_type": "GSO_DataLoader",
        "data": {
            "dataset_config": {
                "image_root": data_path
            },
            "world_size": 1,
            "rank": 0,
            "batch_size": 1,
            "total_num_iter": total_iterations
        },
        
        "renderer": {
            "context_type": "cuda",
            "is_orhto": False
        },
        
        "optimizer": {
            "lr": 0.01,
            "grad_limit": True,
            "grad_limit_values": [0.01, 0.01],
            "grad_limit_iters": [1500]
        },
        
        "use_permute_surface_v": True,
        "permute_surface_v_param": {
            "start_iter": 1500,
            "end_iter": total_iterations,
            "freq": 1000,
            "start_val": 0.01,
            "end_val": 0.001
        },
        
        "verbose": True
    }
    
    return create_hybrid_config_from_dict(base_config, hybrid_overrides)


def create_mesh_fitting_hybrid_config(
    target_mesh_path: str,
    output_path: str = "results/mesh_fitting_hybrid",
    use_warp_rasterization: bool = True,
    total_iterations: int = 15000
) -> DictConfig:
    """Create hybrid configuration for mesh fitting."""
    hybrid_overrides = {
        "use_warp_rasterization": use_warp_rasterization,
        "use_original_geometry": True,
        "use_original_materials": False,
        "use_original_dataloader": False,
        "fallback_to_original": True,
        "total_num_iter": total_iterations,
        "output_path": output_path,
        "verbose": True
    }
    
    base_config = {
        "expr_name": "mesh_fitting_hybrid",
        "fitting_stage": "geometry",
        
        "geometry_type": "TetMeshMultiSphereGeometry",
        "geometry": {
            "initial_mesh_path": "",
            "use_smooth_barrier": True,
            "smooth_barrier_param": {
                "smooth_eng_coeff": 2e-4,
                "barrier_coeff": 2e-4,
                "increase_order_iter": 1000
            },
            "template_surface_sphere_path": "mesh_data/s.1.obj",
            "key_points_file_path": "mesh_data/${expr_name}/${expr_name}.json",
            "tetwild_exec": "/path/to/TetWild/build/TetWild",
            "tetwild_cache_folder": ".tetwild_cache",
            "load_precomputed_tetwild_mesh": False,
            "debug_mode": False
        },
        
        "material_type": None,
        "target_mesh_path": target_mesh_path,
        "mesh_loss_weight": 10000,
        "reg_loss_weight": 0.05,
        "mesh_loss_sample_ratio": 1,
        
        "dataloader_type": "MeshFittingDataLoader",
        "data": {
            "dataset_config": {
                "image_root": "img_data/${expr_name}"
            },
            "world_size": 1,
            "rank": 0,
            "batch_size": 2000,
            "total_num_iter": total_iterations
        },
        
        "renderer": {
            "context_type": "cuda",
            "is_orhto": False
        },
        
        "optimizer": {
            "lr": 0.2,
            "grad_limit": True,
            "grad_limit_values": [0.01, 0.01],
            "grad_limit_iters": [1500]
        },
        
        "use_permute_surface_v": True,
        "permute_surface_v_param": {
            "start_iter": 1500,
            "end_iter": total_iterations,
            "freq": 1000,
            "start_val": 0.01,
            "end_val": 0.001
        },
        
        "verbose": True
    }
    
    return create_hybrid_config_from_dict(base_config, hybrid_overrides)


def create_hybrid_config_from_dict(
    base_config: Dict[str, Any],
    hybrid_overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """Create hybrid configuration from dictionary."""
    hybrid_config = HybridTrainingConfig()
    
    if hybrid_overrides:
        for key, value in hybrid_overrides.items():
            if hasattr(hybrid_config, key):
                setattr(hybrid_config, key, value)
            elif key in hybrid_config.hybrid.__dict__:
                setattr(hybrid_config.hybrid, key, value)
    
    hybrid_dict = OmegaConf.structured(hybrid_config)
    hybrid_dict = OmegaConf.to_container(hybrid_dict, resolve=True)
    merged_config = {**base_config, **hybrid_dict}
    
    return OmegaConf.create(merged_config)


def save_hybrid_config(config: DictConfig, output_path: str):
    """Save hybrid configuration to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)


def load_hybrid_config(config_path: str) -> DictConfig:
    """Load hybrid configuration from file."""
    return OmegaConf.load(config_path)


EXAMPLE_CONFIGS = {
    "gso_hybrid": {
        "description": "GSO dataset training with Warp rasterization",
        "use_warp_rasterization": True,
        "use_original_geometry": True,
        "use_original_materials": True,
        "use_original_dataloader": True
    },
    
    "mesh_fitting_hybrid": {
        "description": "Mesh fitting with Warp rasterization",
        "use_warp_rasterization": True,
        "use_original_geometry": True,
        "use_original_materials": False,
        "use_original_dataloader": False
    },
    
    "full_warp": {
        "description": "Full Warp implementation (experimental)",
        "use_warp_rasterization": True,
        "use_original_geometry": False,
        "use_original_materials": False,
        "use_original_dataloader": False
    },
    
    "original_only": {
        "description": "Original TetSplatting only",
        "use_warp_rasterization": False,
        "use_original_geometry": True,
        "use_original_materials": True,
        "use_original_dataloader": True
    }
}


def get_example_config(config_name: str) -> Dict[str, Any]:
    """Get example configuration by name."""
    if config_name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(EXAMPLE_CONFIGS.keys())}")
    return EXAMPLE_CONFIGS[config_name]
