"""
Pilot Test: CNN Reconstruction and Validation Pipeline

Tests the complete workflow:
1. Generate scenarios
2. Load and normalize weights
3. Train transformer (simplified)
4. Reconstruct CNNs from predictions
5. Finetune and validate
6. Multi-objective ranking
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Import our modules
from cnn_reconstruction import (
    reconstruct_cnn_from_weights,
    finetune_reconstructed_cnn,
    compute_eigenvalues,
    ClassSpecificImageFolder
)
from weight_normalization import (
    LayerWiseNormalizer,
    analyze_weight_distributions,
    compare_normalization_methods
)
from multi_objective_ranking import (
    LossPerformance,
    rank_losses_multi_objective,
    create_ranking_report,
    compute_improvement_rate
)
from generate_scenarios import generate_all_scenarios


def test_mnist_data_loading():
    """Test 1: Verify MNIST data structure"""
    print("\n" + "="*80)
    print("TEST 1: MNIST Data Loading")
    print("="*80)
    
    mnist_root = "/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/SplitMnist"
    
    # Check train directory
    train_dir = Path(mnist_root) / "train"
    test_dir = Path(mnist_root) / "test"
    
    print(f"\nTrain directory: {train_dir}")
    print(f"Exists: {train_dir.exists()}")
    
    if train_dir.exists():
        train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"Train classes: {train_classes}")
    
    print(f"\nTest directory: {test_dir}")
    print(f"Exists: {test_dir.exists()}")
    
    if test_dir.exists():
        test_classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        print(f"Test classes: {test_classes}")
    
    # Test ClassSpecificImageFolder
    try:
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(1)
        ])
        
        # Load data for classes [1,2,3] (drop others)
        dropped = [str(i) for i in range(10) if i not in [1, 2, 3]]
        
        dataset = ClassSpecificImageFolder(
            root=str(test_dir),
            dropped_classes=dropped,
            transform=transform
        )
        
        print(f"\n✓ ClassSpecificImageFolder loaded successfully")
        print(f"  Classes: {dataset.classes}")
        print(f"  Samples: {len(dataset)}")
        
        # Load one sample
        img, label = dataset[0]
        print(f"  Sample shape: {img.shape}")
        print(f"  Sample label: {label}")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        return False
    
    return True


def test_scenario_generation():
    """Test 2: Generate scenarios"""
    print("\n" + "="*80)
    print("TEST 2: Scenario Generation")
    print("="*80)
    
    scenario_dir = "/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Scenario"
    
    try:
        results = generate_all_scenarios(scenario_dir)
        
        print("\n✓ Scenarios generated successfully")
        
        # Verify files exist
        for overlap in [0, 1, 2]:
            overlap_dir = Path(scenario_dir) / f"overlapping_m{overlap}"
            
            train_file = overlap_dir / "train_pairs.npy"
            val_file = overlap_dir / "val_pairs.npy"
            test_file = overlap_dir / "test_pairs.npy"
            
            print(f"\nOverlap m={overlap}:")
            print(f"  Train: {train_file.exists()} ({train_file})")
            print(f"  Val:   {val_file.exists()} ({val_file})")
            print(f"  Test:  {test_file.exists()} ({test_file})")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error generating scenarios: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_normalization():
    """Test 3: Weight normalization"""
    print("\n" + "="*80)
    print("TEST 3: Weight Normalization")
    print("="*80)
    
    zoo_path = "/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Merged zoo.csv"
    
    try:
        # Load sample weights
        print(f"\nLoading weights from {zoo_path}")
        df = pd.read_csv(zoo_path, nrows=1000)  # Load first 1000 for testing
        
        # Extract weight columns (assuming they start at column 11)
        weight_cols = df.columns[11:2475].tolist()
        weights = df[weight_cols].values.astype(np.float32)
        
        print(f"Loaded {len(weights)} weight vectors")
        print(f"Shape: {weights.shape}")
        
        # Analyze distributions
        analyze_weight_distributions(weights[:100], "Sample Weight Distributions")
        
        # Test layer-wise normalization
        print("\nTesting layer-wise normalization...")
        normalizer = LayerWiseNormalizer(method='standard')
        normalized = normalizer.fit_transform(weights)
        
        print(f"\n✓ Normalization successful")
        print(f"  Original mean: {weights.mean():.6f}, std: {weights.std():.6f}")
        print(f"  Normalized mean: {normalized.mean():.6f}, std: {normalized.std():.6f}")
        
        # Test save/load
        save_path = "/tmp/test_normalizer.pkl"
        normalizer.save(save_path)
        
        loaded_normalizer = LayerWiseNormalizer.load(save_path)
        
        # Test inverse transform
        reconstructed = loaded_normalizer.inverse_transform(normalized)
        error = np.abs(reconstructed - weights).mean()
        
        print(f"\n✓ Save/load/inverse successful")
        print(f"  Reconstruction error: {error:.6f}")
        
        return True, weights, normalizer
        
    except Exception as e:
        print(f"\n✗ Error in normalization: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_cnn_reconstruction(weights, normalizer):
    """Test 4: CNN reconstruction and finetuning"""
    print("\n" + "="*80)
    print("TEST 4: CNN Reconstruction and Finetuning")
    print("="*80)
    
    try:
        # Select a random weight vector
        idx = np.random.randint(len(weights))
        weight_vector = weights[idx]
        
        print(f"\nSelected weight vector {idx}")
        print(f"Shape: {weight_vector.shape}")
        
        # Reconstruct CNN
        print("\nReconstructing CNN...")
        cnn = reconstruct_cnn_from_weights(weight_vector, activation="leakyrelu")
        
        print(f"✓ CNN reconstructed")
        print(f"  Parameters: {sum(p.numel() for p in cnn.parameters())}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 1, 28, 28)
        output = cnn(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Compute eigenvalues
        print("\nComputing eigenvalues...")
        eigenvalues = compute_eigenvalues(cnn)
        
        print(f"✓ Eigenvalues computed for {len(eigenvalues)} layers")
        for layer_name, eigs in eigenvalues.items():
            print(f"  {layer_name}: {len(eigs)} eigenvalues")
        
        # Test finetuning (on a small subset)
        print("\nTesting finetuning pipeline...")
        
        task_classes = [1, 2, 3, 4, 5]  # Use classes 1-5
        
        result = finetune_reconstructed_cnn(
            predicted_weights=weight_vector,
            task_classes=task_classes,
            activation="leakyrelu",
            mnist_root="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/SplitMnist",
            n_finetune_epochs=3,  # Reduced for testing
            lr=0.05,
            batch_size=36,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"\n✓ Finetuning completed")
        print(f"  Initial ID accuracy: {result['acc_id_initial']:.2f}%")
        print(f"  Final ID accuracy: {result['acc_id_final']:.2f}%")
        if result['acc_ood_initial'] is not None:
            print(f"  Initial OOD accuracy: {result['acc_ood_initial']:.2f}%")
        
        print(f"\n  Finetuning history:")
        for key, val in result['finetune_history'].items():
            if 'acc_id' in key:
                print(f"    {key}: {val:.2f}%")
        
        return True, result
        
    except Exception as e:
        print(f"\n✗ Error in CNN reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_multi_objective_ranking():
    """Test 5: Multi-objective ranking"""
    print("\n" + "="*80)
    print("TEST 5: Multi-Objective Ranking")
    print("="*80)
    
    # Create sample performances
    performances = [
        LossPerformance(
            loss_name="MSE",
            mse=0.0012,
            initial_acc=72.5,
            final_acc=91.2,
            improvement_rate=3.74,
            finetune_history=[72.5, 78.0, 83.5, 87.0, 89.5, 91.2]
        ),
        LossPerformance(
            loss_name="MAE",
            mse=0.0018,
            initial_acc=78.3,
            final_acc=92.1,
            improvement_rate=2.76,
            finetune_history=[78.3, 82.0, 85.5, 88.0, 90.0, 92.1]
        ),
        LossPerformance(
            loss_name="Huber",
            mse=0.0015,
            initial_acc=75.0,
            final_acc=90.5,
            improvement_rate=3.1,
            finetune_history=[75.0, 79.5, 83.0, 86.0, 88.5, 90.5]
        ),
        LossPerformance(
            loss_name="MAPE",
            mse=0.0025,
            initial_acc=80.0,
            final_acc=91.8,
            improvement_rate=2.36,
            finetune_history=[80.0, 83.0, 86.0, 88.5, 90.0, 91.8]
        ),
    ]
    
    try:
        # Rank losses
        ranked = rank_losses_multi_objective(performances)
        
        # Create report
        df = create_ranking_report(ranked, "Pilot Test Ranking")
        
        print(f"\n✓ Ranking completed")
        print(f"  Winner: {ranked[0][0]}")
        print(f"  Composite score: {ranked[0][1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in ranking: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pilot_test():
    """Run complete pilot test"""
    print("\n" + "="*80)
    print("PILOT TEST: CNN RECONSTRUCTION AND VALIDATION PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: MNIST data
    print("\n" + "→"*40)
    results['mnist_data'] = test_mnist_data_loading()
    
    if not results['mnist_data']:
        print("\n✗ MNIST data test failed. Cannot proceed.")
        return results
    
    # Test 2: Scenario generation
    print("\n" + "→"*40)
    results['scenarios'] = test_scenario_generation()
    
    # Test 3: Weight normalization
    print("\n" + "→"*40)
    success, weights, normalizer = test_weight_normalization()
    results['normalization'] = success
    
    if not success:
        print("\n✗ Normalization test failed. Cannot proceed.")
        return results
    
    # Test 4: CNN reconstruction
    print("\n" + "→"*40)
    success, finetune_result = test_cnn_reconstruction(weights, normalizer)
    results['cnn_reconstruction'] = success
    
    # Test 5: Multi-objective ranking
    print("\n" + "→"*40)
    results['ranking'] = test_multi_objective_ranking()
    
    # Summary
    print("\n" + "="*80)
    print("PILOT TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - Ready for integration!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED - Review errors above")
        print("="*80)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    results = run_full_pilot_test()
    
    # Save results
    output_file = "/tmp/pilot_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
