#!/usr/bin/env python3
"""
Merge Results from Batch 1 and Batch 2
Combines results for ranking and tournament progression
"""

import json
import sys
from pathlib import Path
import pandas as pd

def merge_overlap_results(overlap: int, notebooks_sandbox: Path):
    """Merge batch1 and batch2 results for a specific overlap"""
    
    batch1_file = notebooks_sandbox / f"batch1_overlap{overlap}_results.json"
    batch2_file = notebooks_sandbox / f"batch2_overlap{overlap}_results.json"
    
    # Check if both files exist
    if not batch1_file.exists():
        print(f"⚠ Batch 1 results not found for overlap {overlap}")
        return None
    
    if not batch2_file.exists():
        print(f"⚠ Batch 2 results not found for overlap {overlap}")
        return None
    
    # Load results
    with open(batch1_file, 'r') as f:
        batch1 = json.load(f)
    
    with open(batch2_file, 'r') as f:
        batch2 = json.load(f)
    
    # Merge
    merged = {
        'overlap': overlap,
        'total_losses': len(batch1['losses']) + len(batch2['losses']),
        'batch1_losses': batch1['losses'],
        'batch2_losses': batch2['losses'],
        'all_losses': batch1['losses'] + batch2['losses'],
        'batch1_results': batch1['results'],
        'batch2_results': batch2['results'],
        'all_results': batch1['results'] + batch2['results'],
        'summary': {
            'batch1_successes': batch1['summary']['successes'],
            'batch1_failures': batch1['summary']['failures'],
            'batch2_successes': batch2['summary']['successes'],
            'batch2_failures': batch2['summary']['failures'],
            'total_successes': batch1['summary']['successes'] + batch2['summary']['successes'],
            'total_failures': batch1['summary']['failures'] + batch2['summary']['failures'],
            'total_time_hours': batch1['summary']['total_time_hours'] + batch2['summary']['total_time_hours']
        }
    }
    
    return merged

def main():
    notebooks_sandbox = Path(__file__).parent
    
    print(f"\n{'='*80}")
    print("MERGING BATCH RESULTS")
    print(f"{'='*80}\n")
    
    merged_results = {}
    
    for overlap in [0, 1, 2]:
        print(f"Processing overlap {overlap}...")
        merged = merge_overlap_results(overlap, notebooks_sandbox)
        
        if merged:
            merged_results[overlap] = merged
            
            # Save merged results
            output_file = notebooks_sandbox / f"tiny_overlap{overlap}_merged.json"
            with open(output_file, 'w') as f:
                json.dump(merged, f, indent=2)
            
            print(f"  ✓ Merged {merged['total_losses']} losses")
            print(f"  ✓ Successes: {merged['summary']['total_successes']}")
            print(f"  ✓ Failures: {merged['summary']['total_failures']}")
            print(f"  ✓ Total time: {merged['summary']['total_time_hours']:.1f} hours")
            print(f"  ✓ Saved to: {output_file}\n")
    
    # Create summary across all overlaps
    if merged_results:
        total_experiments = sum(m['total_losses'] for m in merged_results.values())
        total_successes = sum(m['summary']['total_successes'] for m in merged_results.values())
        total_failures = sum(m['summary']['total_failures'] for m in merged_results.values())
        total_time = sum(m['summary']['total_time_hours'] for m in merged_results.values())
        
        summary = {
            'overlaps_completed': list(merged_results.keys()),
            'total_experiments': total_experiments,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'total_time_hours': total_time,
            'per_overlap': {
                str(k): v['summary'] for k, v in merged_results.items()
            }
        }
        
        summary_file = notebooks_sandbox / "tiny_phase1_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"{'='*80}")
        print("PHASE 1 COMPLETE - TINY MODELS")
        print(f"{'='*80}")
        print(f"Total experiments: {total_experiments}")
        print(f"Successes: {total_successes}")
        print(f"Failures: {total_failures}")
        print(f"Total time: {total_time:.1f} hours ({total_time/24:.1f} days)")
        print(f"\nSummary saved to: {summary_file}")
        print(f"\nReady for ranking phase!")
    else:
        print("⚠ No results to merge. Run batch1 and batch2 scripts first.")
        sys.exit(1)

if __name__ == "__main__":
    main()
