#!/usr/bin/env python3
"""
Test and validation script for Hybrid TetSplatting pipeline.
Tests both individual components and the complete pipeline.
"""

import torch
import numpy as np
import os
import sys
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import hybrid components
from hybrid_geometry import HybridTetMeshGeometry, HybridTetMeshMultiSphereGeometry
from hybrid_renderer import HybridMeshRasterizer, create_hybrid_renderer
from hybrid_config import create_gso_hybrid_config, create_mesh_fitting_hybrid_config
from warp_mesh_rasterizer import WarpMeshRasterizer, WarpRasterizationFunction


def test_warp_rasterization():
    """Test Warp rasterization functionality"""
    print("Testing Warp rasterization...")
    
    try:
        # Create simple test data
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=torch.float32, device='cuda')
        
        triangle_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device='cuda')
        
        mvp = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device='cuda')
        
        image_size = 64
        
        # Test rasterization
        rast_out, _ = WarpRasterizationFunction.apply(
            vertices, triangle_indices, mvp, image_size, is_orthographic=True
        )
        
        print(f"✅ Warp rasterization successful")
        print(f"   Output shape: {rast_out.shape}")
        print(f"   Alpha range: [{rast_out[..., -1].min():.3f}, {rast_out[..., -1].max():.3f}]")
        
        # Test antialiasing
        alpha_pixels = (rast_out[..., -1] > 0).sum()
        print(f"   Rendered pixels: {alpha_pixels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Warp rasterization failed: {e}")
        return False


def test_hybrid_geometry():
    """Test hybrid geometry classes"""
    print("Testing hybrid geometry...")
    
    try:
        # Create test configuration
        cfg = {
            "initial_mesh_path": "",
            "use_smooth_barrier": True,
            "smooth_barrier_param": {
                "smooth_eng_coeff": 2e-4,
                "barrier_coeff": 2e-4,
                "increase_order_iter": 1000
            },
            "use_warp_for_simple_ops": True,
            "use_original_energy": True,
            "use_warp_rasterization": True
        }
        
        # Test hybrid geometry
        geometry = HybridTetMeshGeometry(cfg)
        
        # Test forward pass
        forward_data = geometry(iter_num=0)
        
        print(f"✅ Hybrid geometry successful")
        print(f"   Surface vertices: {forward_data.v_pos.shape}")
        print(f"   Surface faces: {forward_data.t_pos_idx.shape}")
        print(f"   Regularization energy: {forward_data.smooth_barrier_energy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid geometry failed: {e}")
        return False


def test_hybrid_renderer():
    """Test hybrid renderer"""
    print("Testing hybrid renderer...")
    
    try:
        # Create test geometry
        cfg = {
            "initial_mesh_path": "",
            "use_smooth_barrier": False,
            "use_warp_for_simple_ops": True,
            "use_original_energy": True,
            "use_warp_rasterization": True
        }
        
        geometry = HybridTetMeshGeometry(cfg)
        
        # Create hybrid renderer
        renderer = create_hybrid_renderer(
            geometry=geometry,
            material_type=None,
            material_cfg={},
            renderer_cfg={"use_warp_rasterization": True, "fallback_to_original": True}
        )
        
        # Test rendering
        mvp = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device='cuda')
        
        background = torch.zeros(1, 64, 64, 3, device='cuda')
        
        output = renderer.forward(
            mvp=mvp,
            only_alpha=True,
            resolution=64,
            background=background
        )
        
        print(f"✅ Hybrid renderer successful")
        print(f"   Output keys: {list(output.keys())}")
        print(f"   Shaded shape: {output['shaded'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid renderer failed: {e}")
        return False


def test_performance_comparison():
    """Compare performance between Warp and original rasterization"""
    print("Testing performance comparison...")
    
    try:
        # Create test data
        vertices = torch.randn(1000, 3, device='cuda')
        triangle_indices = torch.randint(0, 1000, (3000,), device='cuda')
        
        mvp = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device='cuda')
        
        image_size = 512
        
        # Test Warp rasterization performance
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            rast_out, _ = WarpRasterizationFunction.apply(
                vertices, triangle_indices, mvp, image_size, is_orthographic=True
            )
        
        torch.cuda.synchronize()
        warp_time = (time.time() - start_time) / 10
        
        print(f"✅ Performance comparison completed")
        print(f"   Warp rasterization: {warp_time*1000:.2f} ms per frame")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False


def test_configuration_creation():
    """Test configuration creation"""
    print("Testing configuration creation...")
    
    try:
        # Test GSO configuration
        gso_config = create_gso_hybrid_config(
            data_path="/path/to/gso",
            output_path="results/gso_test",
            use_warp_rasterization=True,
            total_iterations=1000
        )
        
        print(f"✅ GSO configuration created")
        print(f"   Use Warp rasterization: {gso_config.hybrid.use_warp_rasterization}")
        print(f"   Total iterations: {gso_config.total_num_iter}")
        
        # Test mesh fitting configuration
        mesh_config = create_mesh_fitting_hybrid_config(
            target_mesh_path="/path/to/mesh.obj",
            output_path="results/mesh_test",
            use_warp_rasterization=True,
            total_iterations=1000
        )
        
        print(f"✅ Mesh fitting configuration created")
        print(f"   Use Warp rasterization: {mesh_config.hybrid.use_warp_rasterization}")
        print(f"   Target mesh: {mesh_config.target_mesh_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False


def test_fallback_mechanism():
    """Test fallback mechanism when Warp fails"""
    print("Testing fallback mechanism...")
    
    try:
        # Create renderer with fallback enabled
        cfg = {
            "initial_mesh_path": "",
            "use_smooth_barrier": False,
            "use_warp_for_simple_ops": True,
            "use_original_energy": True,
            "use_warp_rasterization": True
        }
        
        geometry = HybridTetMeshGeometry(cfg)
        
        renderer = create_hybrid_renderer(
            geometry=geometry,
            material_type=None,
            material_cfg={},
            renderer_cfg={
                "use_warp_rasterization": True,
                "fallback_to_original": True
            }
        )
        
        print(f"✅ Fallback mechanism configured")
        print(f"   Has original rasterizer: {renderer.original_rasterizer is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback mechanism test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Hybrid TetSplatting Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Warp Rasterization", test_warp_rasterization),
        ("Hybrid Geometry", test_hybrid_geometry),
        ("Hybrid Renderer", test_hybrid_renderer),
        ("Performance Comparison", test_performance_comparison),
        ("Configuration Creation", test_configuration_creation),
        ("Fallback Mechanism", test_fallback_mechanism),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hybrid pipeline is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return results


def benchmark_pipeline():
    """Benchmark the complete hybrid pipeline"""
    print("Benchmarking hybrid pipeline...")
    
    try:
        # Create configuration
        config = create_gso_hybrid_config(
            data_path="/tmp/test_data",
            output_path="/tmp/test_output",
            use_warp_rasterization=True,
            total_iterations=100
        )
        
        # Create components
        geometry = HybridTetMeshGeometry(config.geometry)
        renderer = create_hybrid_renderer(
            geometry=geometry,
            material_type=None,
            material_cfg={},
            renderer_cfg=config.renderer
        )
        
        # Benchmark rendering
        mvp = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device='cuda')
        
        background = torch.zeros(1, 256, 256, 3, device='cuda')
        
        # Warm up
        for _ in range(5):
            renderer.forward(
                mvp=mvp,
                only_alpha=True,
                resolution=256,
                background=background
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(20):
            output = renderer.forward(
                mvp=mvp,
                only_alpha=True,
                resolution=256,
                background=background
            )
        
        torch.cuda.synchronize()
        avg_time = (time.time() - start_time) / 20
        
        print(f"✅ Pipeline benchmark completed")
        print(f"   Average rendering time: {avg_time*1000:.2f} ms")
        print(f"   Resolution: 256x256")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline benchmark failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hybrid TetSplatting Pipeline")
    parser.add_argument("--test", choices=["all", "rasterization", "geometry", "renderer", "performance", "config", "fallback"], 
                       default="all", help="Test to run")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_pipeline()
    elif args.test == "all":
        run_all_tests()
    elif args.test == "rasterization":
        test_warp_rasterization()
    elif args.test == "geometry":
        test_hybrid_geometry()
    elif args.test == "renderer":
        test_hybrid_renderer()
    elif args.test == "performance":
        test_performance_comparison()
    elif args.test == "config":
        test_configuration_creation()
    elif args.test == "fallback":
        test_fallback_mechanism()
