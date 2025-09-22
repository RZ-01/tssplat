"""
Shape fitting trainer using NVIDIA Warp.
Replaces multi-view reconstruction with direct shape fitting to target mesh.
"""

import torch
import warp as wp
import os
from typing import Dict
from dataclasses import dataclass
import yaml
from tqdm import tqdm

from warp_tetmesh import WarpTetMesh, WarpBarrierEnergy, load_tetmesh_from_veg
from geometric_loss import WarpGeometricLoss, load_target_mesh


@dataclass
class ShapeFittingConfig:
    """Configuration for shape fitting"""
    # Mesh paths
    initial_tetmesh_path: str
    target_mesh_path: str
    output_path: str
    
    # Training parameters
    num_iterations: int = 2000
    learning_rate: float = 0.01
    
    # Loss weights
    geometric_loss_weight: float = 1.0
    barrier_loss_weight: float = 1e-4
    smoothness_loss_weight: float = 1e-4
    
    # Barrier energy parameters
    barrier_order: int = 2
    barrier_order_increase_iter: int = 1000
    
    # Optimization parameters
    grad_clip_value: float = 0.01
    
    # Logging
    save_interval: int = 100
    log_interval: int = 10


class ShapeFittingTrainer:
    """Main trainer for shape fitting using tetrahedral meshes"""
    
    def __init__(self, config: ShapeFittingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Warp
        wp.init()
        
        # Load meshes
        self._load_meshes()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup output directory
        os.makedirs(config.output_path, exist_ok=True)
        
    def _load_meshes(self):
        """Load tetrahedral mesh and target mesh"""
        print("Loading meshes...")
        
        # Load tetrahedral mesh
        self.tetmesh = load_tetmesh_from_veg(self.config.initial_tetmesh_path)
            
        # Convert to PyTorch tensors
        self.tet_vertices = torch.from_numpy(self.tetmesh.vertices).float().to(self.device)
        self.tet_vertices.requires_grad_(True)
        self.tet_vertices.retain_grad()  # Ensure gradients are retained
        
        self.tet_elements = torch.from_numpy(self.tetmesh.elements).long().to(self.device)
        self.rest_matrices = torch.from_numpy(self.tetmesh.rest_matrices).float().to(self.device)
        
        self.surface_vertex_indices = torch.from_numpy(self.tetmesh.surface_vertices).long().to(self.device)
        
        print(f"Loaded tetmesh: {len(self.tetmesh.vertices)} vertices, {len(self.tetmesh.elements)} tetrahedra")
        print(f"Surface vertices: {len(self.tetmesh.surface_vertices)}")
        
        # Load target mesh
        self.target_mesh = load_target_mesh(self.config.target_mesh_path)
        self.geometric_loss = WarpGeometricLoss(self.target_mesh, self.config.geometric_loss_weight)
        self.geometric_loss.to(self.device)
        
        print(f"Loaded target mesh: {len(self.target_mesh.vertices)} vertices, {len(self.target_mesh.faces)} faces")
        
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.Adam([self.tet_vertices], lr=self.config.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.num_iterations,
            eta_min=self.config.learning_rate * 0.01
        )
        
    def get_surface_vertices(self) -> torch.Tensor:
        """Get surface vertex positions"""
        return self.tet_vertices[self.surface_vertex_indices]
        
    def compute_barrier_energy(self, iteration: int) -> torch.Tensor:
        """Compute barrier energy for current tetmesh state"""
        # Determine barrier order
        order = self.config.barrier_order
        if iteration > self.config.barrier_order_increase_iter:
            order = 4
            
        # Compute barrier energy
        barrier_energy = WarpBarrierEnergy.apply(
            self.tet_vertices,
            self.tet_elements.flatten(),  # Flatten for Warp kernel
            self.rest_matrices,
            self.config.smoothness_loss_weight,
            self.config.barrier_loss_weight,
            order
        )
        
        return barrier_energy
        
    def compute_geometric_loss(self) -> torch.Tensor:
        """Compute geometric loss between surface and target mesh"""
        surface_vertices = self.get_surface_vertices()
        return self.geometric_loss.compute_loss(surface_vertices)
        
    def compute_total_loss(self, iteration: int) -> Dict[str, torch.Tensor]:
        """Compute total loss and individual loss components"""
        # Geometric loss (main data term)
        geometric_loss = self.compute_geometric_loss()
        
        # Barrier energy (regularization)
        barrier_loss = self.compute_barrier_energy(iteration)
        
        # Total loss
        total_loss = geometric_loss + barrier_loss
        
        return {
            'total': total_loss,
            'geometric': geometric_loss,
            'barrier': barrier_loss
        }
        
    def save_current_mesh(self, iteration: int, prefix: str = ""):
        """Save current mesh state"""
        # Update tetmesh vertices
        vertices_np = self.tet_vertices.detach().cpu().numpy()
        
        # Create new tetmesh with updated vertices
        current_mesh = WarpTetMesh(
            vertices_np, 
            self.tetmesh.elements,
            self.tetmesh.surface_vertices,
            self.tetmesh.surface_faces
        )
        
        # Save surface mesh
        filename = f"{prefix}mesh_{iteration:05d}.obj"
        filepath = os.path.join(self.config.output_path, filename)
        current_mesh.export_mesh(filepath)
        
        return filepath
        
    def train(self):
        """Main training loop"""
        print("Starting shape fitting training...")
        print(f"Device: {self.device}")
        print(f"Iterations: {self.config.num_iterations}")
        
        best_loss = float('inf')
        best_iteration = 0
        
        for iteration in tqdm(range(self.config.num_iterations), desc="Training"):
            # Forward pass
            losses = self.compute_total_loss(iteration)
            total_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_([self.tet_vertices], self.config.grad_clip_value)
            
            # Optimization step
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            if iteration % self.config.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                tqdm.write(
                    f"Iter {iteration:4d}: "
                    f"Total={total_loss.item():.6f}, "
                    f"Geo={losses['geometric'].item():.6f}, "
                    f"Barrier={losses['barrier'].item():.6f}, "
                    f"LR={current_lr:.2e}"
                )
                
            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_iteration = iteration
                self.save_current_mesh(iteration, "best_")
                
            # Periodic saves
            if iteration % self.config.save_interval == 0:
                self.save_current_mesh(iteration)
                
        print(f"Training completed!")
        print(f"Best loss: {best_loss:.6f} at iteration {best_iteration}")
        
        # Final save
        final_path = self.save_current_mesh(self.config.num_iterations - 1, "final_")
        print(f"Final mesh saved to: {final_path}")
        
        return {
            'best_loss': best_loss,
            'best_iteration': best_iteration,
            'final_mesh_path': final_path
        }


def load_config(config_path: str) -> ShapeFittingConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ShapeFittingConfig(**config_dict)


def create_example_config(output_path: str = "config/shape_fitting_example.yaml"):
    """Create an example configuration file"""
    config = {
        'initial_tetmesh_path': 'mesh_data/initial_sphere.veg',
        'target_mesh_path': 'mesh_data/target_shape.obj', 
        'output_path': 'results/shape_fitting',
        'num_iterations': 2000,
        'learning_rate': 0.01,
        'geometric_loss_weight': 1.0,
        'barrier_loss_weight': 1e-4,
        'smoothness_loss_weight': 1e-4,
        'barrier_order': 2,
        'barrier_order_increase_iter': 1000,
        'grad_clip_value': 0.01,
        'save_interval': 100,
        'log_interval': 10
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Example config saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shape fitting with tetrahedral meshes")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-example-config", action="store_true", 
                        help="Create example configuration file")
    
    args = parser.parse_args()
    
    if args.create_example_config:
        create_example_config()
    elif args.config:
        config = load_config(args.config)
        trainer = ShapeFittingTrainer(config)
        results = trainer.train()
        print("Training results:", results)
    else:
        print("Please provide --config or use --create-example-config")
        print("Example usage:")
        print("  python shape_fitting_trainer.py --create-example-config")
        print("  python shape_fitting_trainer.py --config config/shape_fitting_example.yaml")