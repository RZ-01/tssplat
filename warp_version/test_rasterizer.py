#!/usr/bin/env python3
"""
Test script for the improved Warp mesh rasterizer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from warp_mesh_rasterizer import WarpMeshRasterizer, WarpRasterizationFunction

def create_simple_triangle():
    """Create a simple triangle for testing"""
    # Define vertices of a triangle
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],    # Bottom left
        [1.0, 0.0, 0.0],    # Bottom right  
        [0.5, 1.0, 0.0],    # Top center
    ], dtype=torch.float32, device='cuda')
    
    # Define triangle indices
    triangle_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device='cuda')
    
    return vertices, triangle_indices

def create_simple_mvp():
    """Create a simple MVP matrix for testing"""
    # Simple orthographic projection
    mvp = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32, device='cuda')
    
    return mvp

def test_basic_rasterization():
    """Test basic rasterization functionality"""
    print("Testing basic rasterization...")
    
    # Create test data
    vertices, triangle_indices = create_simple_triangle()
    mvp = create_simple_mvp()
    image_size = 64
    
    # Test the rasterization function directly
    rast_out, _ = WarpRasterizationFunction.apply(
        vertices, triangle_indices, mvp, image_size, is_orthographic=True
    )
    
    print(f"Rasterization output shape: {rast_out.shape}")
    print(f"Alpha range: [{rast_out[..., -1].min():.3f}, {rast_out[..., -1].max():.3f}]")
    
    # Check if we got some pixels
    alpha_pixels = (rast_out[..., -1] > 0).sum()
    print(f"Number of rendered pixels: {alpha_pixels}")
    
    return rast_out

def test_antialiasing():
    """Test antialiasing functionality"""
    print("\nTesting antialiasing...")
    
    vertices, triangle_indices = create_simple_triangle()
    mvp = create_simple_mvp()
    image_size = 64
    
    # Rasterize
    rast_out, _ = WarpRasterizationFunction.apply(
        vertices, triangle_indices, mvp, image_size, is_orthographic=True
    )
    
    alpha = rast_out[..., -1]
    
    # Check edge pixels for antialiasing
    edge_pixels = []
    for i in range(1, image_size - 1):
        for j in range(1, image_size - 1):
            if alpha[i, j] > 0:
                # Check if this is an edge pixel
                neighbors = [
                    alpha[i-1, j], alpha[i+1, j],
                    alpha[i, j-1], alpha[i, j+1]
                ]
                if any(a == 0 for a in neighbors):
                    edge_pixels.append(alpha[i, j].item())
    
    if edge_pixels:
        print(f"Edge pixel alpha values: {edge_pixels[:10]}...")  # Show first 10
        print(f"Edge alpha range: [{min(edge_pixels):.3f}, {max(edge_pixels):.3f}]")
    else:
        print("No edge pixels found")

def visualize_rasterization():
    """Visualize the rasterization result"""
    print("\nVisualizing rasterization...")
    
    vertices, triangle_indices = create_simple_triangle()
    mvp = create_simple_mvp()
    image_size = 128
    
    # Rasterize
    rast_out, _ = WarpRasterizationFunction.apply(
        vertices, triangle_indices, mvp, image_size, is_orthographic=True
    )
    
    alpha = rast_out[..., -1].cpu().numpy()
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(alpha, cmap='gray')
    plt.title('Alpha Channel')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(alpha > 0, cmap='gray')
    plt.title('Binary Mask')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('/Users/rayallen/Documents/Projects/tssplat/warp_version/rasterization_test.png')
    plt.close()
    
    print("Visualization saved to rasterization_test.png")

def test_performance():
    """Test rasterization performance"""
    print("\nTesting performance...")
    
    vertices, triangle_indices = create_simple_triangle()
    mvp = create_simple_mvp()
    image_size = 512
    
    # Warm up
    for _ in range(5):
        _ = WarpRasterizationFunction.apply(
            vertices, triangle_indices, mvp, image_size, is_orthographic=True
        )
    
    # Time the rasterization
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(100):
        _ = WarpRasterizationFunction.apply(
            vertices, triangle_indices, mvp, image_size, is_orthographic=True
        )
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 100.0  # Average time per call
    
    print(f"Average rasterization time: {elapsed_time:.2f} ms")
    print(f"Resolution: {image_size}x{image_size}")

if __name__ == "__main__":
    print("Warp Mesh Rasterizer Test")
    print("=" * 40)
    
    try:
        # Test basic functionality
        rast_out = test_basic_rasterization()
        
        # Test antialiasing
        test_antialiasing()
        
        # Visualize results
        visualize_rasterization()
        
        # Test performance
        test_performance()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
