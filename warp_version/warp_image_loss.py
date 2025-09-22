"""
Image loss functions for Warp-based TetSplatting.
Implements MSE, L1, and depth losses for multi-view reconstruction.
"""

import torch
import torch.nn as nn
from typing import Dict


class WarpImageLoss:
    """Image loss functions for TetSplatting - directly copied from original trainer logic"""
    
    def __init__(self):
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def compute_image_loss(self, 
                          rendered_image: torch.Tensor,
                          target_image: torch.Tensor,
                          fitting_stage: str = "geometry",
                          loss_type: str = "mse") -> torch.Tensor:
        """
        Compute image reconstruction loss
        
        Args:
            rendered_image: Rendered image [H, W, C]
            target_image: Target image [H, W, C] 
            fitting_stage: "geometry" or "texture"
            loss_type: "mse" or "l1"
        
        Returns:
            Image loss tensor
        """
        if fitting_stage == "geometry":
            # For geometry stage, only use alpha channel
            rendered_alpha = rendered_image[..., -1]
            target_alpha = target_image[..., -1]
            
            if loss_type == "mse":
                img_loss = self.mse_loss(rendered_alpha, target_alpha)
            else:
                img_loss = self.l1_loss(rendered_alpha, target_alpha)
        else:
            # For texture stage, use RGB channels
            rendered_rgb = rendered_image[..., :3]
            target_rgb = target_image[..., :3]
            
            if loss_type == "mse":
                img_loss = self.mse_loss(rendered_rgb, target_rgb)
            else:
                img_loss = self.l1_loss(rendered_rgb, target_rgb)
        
        # Scale factor from original trainer
        img_loss *= 20
        
        return img_loss
    
    def compute_depth_loss(self,
                          rendered_depth: torch.Tensor,
                          target_depth: torch.Tensor,
                          alpha_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute depth loss - directly copied from original trainer
        
        Args:
            rendered_depth: Rendered depth [H, W, 1]
            target_depth: Target depth [H, W, 1]
            alpha_mask: Alpha mask [H, W, 1]
        
        Returns:
            Depth loss tensor
        """
        # Apply alpha mask to both depth maps
        masked_rendered = rendered_depth * alpha_mask
        masked_target = target_depth * alpha_mask
        
        depth_loss = self.mse_loss(masked_rendered, masked_target)
        
        # Scale factor from original trainer
        depth_loss *= 100
        
        return depth_loss
    
    def compute_total_loss(self,
                          rendered_outputs: Dict[str, torch.Tensor],
                          target_batch: Dict[str, torch.Tensor],
                          fitting_stage: str = "geometry",
                          fit_depth: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining image and regularization terms
        - directly copied logic from original trainer
        
        Args:
            rendered_outputs: Dictionary with rendered outputs
            target_batch: Dictionary with target data
            fitting_stage: "geometry" or "texture"
            fit_depth: Whether to include depth loss
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Image loss
        img_loss = self.compute_image_loss(
            rendered_outputs["shaded"],
            target_batch["img"][0],  # Remove batch dimension
            fitting_stage=fitting_stage,
            loss_type="mse" if fitting_stage == "geometry" else "l1"
        )
        losses["img_loss"] = img_loss
        
        # Depth loss
        if fit_depth:
            depth_loss = self.compute_depth_loss(
                rendered_outputs["d"],
                target_batch["d"][0],  # Remove batch dimension  
                target_batch["img"][0][..., -1:]  # Alpha channel
            )
            losses["depth_loss"] = depth_loss
            img_loss += depth_loss
        
        # Regularization loss
        reg_loss = 0.0
        if fitting_stage == "geometry":
            reg_loss = rendered_outputs["geo_regularization"]
        losses["reg_loss"] = reg_loss
        
        # Total loss (same weights as original trainer)
        total_loss = img_loss * 100 + reg_loss
        losses["total_loss"] = total_loss
        
        return losses


class WarpLPIPSLoss:
    """LPIPS loss for perceptual similarity (optional enhancement)"""
    
    def __init__(self):
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
            self.available = True
        except ImportError:
            print("Warning: LPIPS not available, falling back to L1 loss")
            self.lpips_fn = nn.L1Loss()
            self.available = False
    
    def compute_lpips_loss(self, 
                          rendered_image: torch.Tensor,
                          target_image: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss
        
        Args:
            rendered_image: Rendered RGB image [H, W, 3]
            target_image: Target RGB image [H, W, 3]
        
        Returns:
            LPIPS loss tensor
        """
        if not self.available:
            return self.lpips_fn(rendered_image, target_image)
        
        # Convert to format expected by LPIPS [1, 3, H, W]
        rendered = rendered_image.permute(2, 0, 1).unsqueeze(0)
        target = target_image.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize to [-1, 1] range
        rendered = rendered * 2.0 - 1.0
        target = target * 2.0 - 1.0
        
        return self.lpips_fn(rendered, target)


def create_loss_function(loss_type: str = "mse") -> WarpImageLoss:
    """Factory function to create loss function"""
    return WarpImageLoss()


# For testing
if __name__ == "__main__":
    # Test loss functions
    loss_fn = WarpImageLoss()
    
    # Create dummy data
    rendered = torch.randn(512, 512, 4)  # RGBA
    target = torch.randn(512, 512, 4)    # RGBA
    
    # Test geometry stage loss (alpha only)
    geo_loss = loss_fn.compute_image_loss(rendered, target, fitting_stage="geometry")
    print(f"Geometry loss: {geo_loss.item()}")
    
    # Test texture stage loss (RGB)
    tex_loss = loss_fn.compute_image_loss(rendered, target, fitting_stage="texture")
    print(f"Texture loss: {tex_loss.item()}")
    
    # Test depth loss
    rendered_depth = torch.randn(512, 512, 1)
    target_depth = torch.randn(512, 512, 1)
    alpha_mask = torch.randn(512, 512, 1)
    
    depth_loss = loss_fn.compute_depth_loss(rendered_depth, target_depth, alpha_mask)
    print(f"Depth loss: {depth_loss.item()}")
    
    print("Loss functions test completed!")