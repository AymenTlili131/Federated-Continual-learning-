"""
Generate overlap-based scenarios for meta-learning experiments

Creates train/val/test splits based on task overlap (m=0,1,2)
WITHOUT epoch or activation specificity - those are training variables
"""

import numpy as np
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import json


def generate_overlap_scenarios(overlap: int, save_dir: str = "./data/Scenario"):
    """
    Generate scenarios for a specific overlap level
    
    Criteria (IMPROVED for better test set coverage):
    - Train: 70% of pairs (diverse task sizes, balanced)
    - Val: 15% of pairs (same distribution as train)
    - Test: 15% of pairs (challenging OOD scenarios)
    
    Test set criteria (multi-faceted OOD challenge):
    1. Extreme size differences (|len(t1) - len(t2)| >= 4)
    2. Contains rare digits (0, 1, or 9)
    3. Very small tasks (len < 3) or very large (len > 7)
    4. Asymmetric overlaps (one task much larger than overlap)
    
    Args:
        overlap: Number of overlapping classes (0, 1, or 2)
        save_dir: Directory to save scenarios
    """
    print(f"\n{'='*60}")
    print(f"Generating scenarios for overlap m={overlap}")
    print(f"{'='*60}")
    
    all_pairs = []
    digits = list(range(10))
    
    # Generate all valid task pairs with specified overlap
    for k1 in tqdm(range(2, 10), desc=f"Task size 1"):
        S1 = list(combinations(digits, k1))
        for pair1 in S1:
            for k2 in range(2, 10):
                S2 = list(combinations(digits, k2))
                for pair2 in S2:
                    # Check: different pairs, correct overlap
                    if pair1 != pair2 and len(set(pair1) & set(pair2)) == overlap:
                        pair_list = [list(pair1), list(pair2)]
                        if pair_list not in all_pairs:
                            all_pairs.append(pair_list)
    
    print(f"Total task pairs generated: {len(all_pairs)}")
    
    # Compute OOD scores for each pair (higher = more challenging)
    def compute_ood_score(task1, task2):
        """Compute how challenging/OOD a task pair is"""
        score = 0
        
        # 1. Extreme size difference (max 4 points)
        size_diff = abs(len(task1) - len(task2))
        score += min(size_diff, 4)
        
        # 2. Contains rare digits 0, 1, 9 (1 point each)
        rare_digits = {0, 1, 9}
        if any(d in rare_digits for d in task1 + task2):
            score += 2
        
        # 3. Very small or very large tasks (2 points)
        if len(task1) <= 2 or len(task2) <= 2:
            score += 2
        if len(task1) >= 8 or len(task2) >= 8:
            score += 2
        
        # 4. Asymmetric overlap (overlap much smaller than tasks)
        if overlap > 0:
            min_size = min(len(task1), len(task2))
            if overlap / min_size < 0.3:  # Overlap < 30% of smaller task
                score += 2
        
        # 5. Contains both even and odd digits (diversity)
        all_digits = set(task1 + task2)
        has_even = any(d % 2 == 0 for d in all_digits)
        has_odd = any(d % 2 == 1 for d in all_digits)
        if has_even and has_odd:
            score += 1
        
        return score
    
    # Score all pairs
    scored_pairs = [(pair, compute_ood_score(pair[0], pair[1])) for pair in all_pairs]
    scored_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by OOD score
    
    # Split: top 15% → test, next 15% → val, rest → train
    n_total = len(scored_pairs)
    n_test = max(int(n_total * 0.15), 1)
    n_val = max(int(n_total * 0.15), 1)
    
    test_pairs = [pair for pair, _ in scored_pairs[:n_test]]
    val_pairs = [pair for pair, _ in scored_pairs[n_test:n_test + n_val]]
    train_pairs = [pair for pair, _ in scored_pairs[n_test + n_val:]]
    
    print(f"\nSplit statistics:")
    print(f"  Train pairs: {len(train_pairs)} ({100*len(train_pairs)/n_total:.1f}%)")
    print(f"  Val pairs:   {len(val_pairs)} ({100*len(val_pairs)/n_total:.1f}%)")
    print(f"  Test pairs:  {len(test_pairs)} ({100*len(test_pairs)/n_total:.1f}%) - OOD challenge")
    
    # Print test set characteristics
    if test_pairs:
        test_scores = [compute_ood_score(p[0], p[1]) for p in test_pairs]
        print(f"\n  Test set OOD scores: min={min(test_scores)}, max={max(test_scores)}, avg={np.mean(test_scores):.1f}")
    
    # Save scenarios
    scenario_dir = Path(save_dir) / f"overlapping_m{overlap}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(scenario_dir / "train_pairs.npy", np.array(train_pairs, dtype=object))
    np.save(scenario_dir / "val_pairs.npy", np.array(val_pairs, dtype=object))
    np.save(scenario_dir / "test_pairs.npy", np.array(test_pairs, dtype=object))
    
    # Save metadata
    metadata = {
        'overlap': overlap,
        'total_pairs': len(all_pairs),
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
        'test_pairs': len(test_pairs),
        'criteria': {
            'train': 'Different task sizes',
            'val': 'Same task sizes',
            'test': 'Contains class 0 (OOD)'
        }
    }
    
    with open(scenario_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved to: {scenario_dir}")
    print(f"  - train_pairs.npy")
    print(f"  - val_pairs.npy")
    print(f"  - test_pairs.npy")
    print(f"  - metadata.json")
    
    return train_pairs, val_pairs, test_pairs, metadata


def generate_all_scenarios(save_dir: str = "./data/Scenario"):
    """Generate scenarios for all overlap levels (0, 1, 2)"""
    print("\n" + "="*60)
    print("GENERATING ALL OVERLAP SCENARIOS")
    print("="*60)
    
    results = {}
    
    for overlap in [0, 1, 2]:
        train, val, test, meta = generate_overlap_scenarios(overlap, save_dir)
        results[f'm{overlap}'] = {
            'train': len(train),
            'val': len(val),
            'test': len(test),
            'metadata': meta
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for overlap_key, data in results.items():
        print(f"\n{overlap_key}:")
        print(f"  Train: {data['train']:,} pairs")
        print(f"  Val:   {data['val']:,} pairs")
        print(f"  Test:  {data['test']:,} pairs")
        print(f"  Total: {data['train'] + data['val'] + data['test']:,} pairs")
    
    total_experiments = sum(
        data['train'] + data['val'] + data['test'] 
        for data in results.values()
    )
    print(f"\nTotal unique task pairs across all overlaps: {total_experiments:,}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Get save directory from command line or use default
    save_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Scenario"
    
    print(f"Save directory: {save_dir}")
    
    # Generate all scenarios
    results = generate_all_scenarios(save_dir)
    
    print("\n✅ Scenario generation complete!")
