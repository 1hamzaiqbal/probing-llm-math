#!/usr/bin/env python3
"""
Train success probes with token length filtering.
Filters out samples with 1000+ tokens to avoid confounding.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_filter_data(data_file, audit_file, max_tokens=1000):
    """Load probe data and filter by token count."""
    
    # Load probe data
    print(f"\nLoading: {data_file}")
    data = torch.load(data_file, weights_only=False)
    
    # Debug: show what we loaded
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Dict keys: {list(data.keys())}")
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'not a dict'}")
    
    # Handle nested format: {'data': [...], 'metadata': {...}}
    if isinstance(data, dict) and 'data' in data and 'metadata' in data:
        print("Detected nested format with 'data' and 'metadata' keys")
        actual_data = data['data']  # This is likely a list of dicts
        metadata = data['metadata']
        print(f"Metadata: {metadata}")
        print(f"Inner data type: {type(actual_data)}, length: {len(actual_data) if hasattr(actual_data, '__len__') else 'N/A'}")
        data = actual_data  # Use the inner data
    
    # Load audit file for token counts
    token_counts = None
    if audit_file and os.path.exists(audit_file):
        print(f"\nLoading audit: {audit_file}")
        audit_df = pd.read_csv(audit_file)
        print(f"Audit columns: {list(audit_df.columns)}")
        print(f"Audit rows: {len(audit_df)}")
        if 'num_tokens' in audit_df.columns:
            token_counts = audit_df['num_tokens'].values
            print(f"Token counts range: {token_counts.min()} - {token_counts.max()}")
    else:
        print(f"Warning: Audit file not found: {audit_file}")
    
    # Check format (new vs old)
    if isinstance(data, dict) and 'activations_0pct' in data:
        # New format (stacked tensors)
        n_samples = len(data['labels'])
        print(f"\nNew format detected: {n_samples} samples")
        
        if token_counts is None:
            token_counts = np.zeros(n_samples)
            print("Warning: No token counts, using all samples")
        
        # Ensure token_counts matches data length
        if len(token_counts) != n_samples:
            print(f"Warning: Token count mismatch ({len(token_counts)} vs {n_samples}), using all samples")
            token_counts = np.zeros(n_samples)
        
        # Filter by token count
        mask = token_counts < max_tokens
        n_filtered = mask.sum()
        n_removed = n_samples - n_filtered
        print(f"Filtering: {n_samples} -> {n_filtered} samples (removed {n_removed} with {max_tokens}+ tokens)")
        
        # Apply filter to tensors
        filtered_data = {
            'activations_0pct': data['activations_0pct'][mask],
            'activations_100pct': data['activations_100pct'][mask],
            'labels': np.array(data['labels'])[mask] if not isinstance(data['labels'], np.ndarray) else data['labels'][mask],
            'token_counts': token_counts[mask],
            'format': 'new'
        }
        
        # Handle 50pct if present
        if 'activations_50pct' in data and data['activations_50pct'] is not None:
            if isinstance(data['activations_50pct'], torch.Tensor):
                filtered_data['activations_50pct'] = data['activations_50pct'][mask]
        
        # Copy metadata
        filtered_data['metadata'] = data.get('metadata', {})
        
        return filtered_data
        
    elif isinstance(data, list) and len(data) > 0:
        # Old format (list of dicts)
        n_samples = len(data)
        print(f"\nOld format (list) detected: {n_samples} samples")
        
        if token_counts is None:
            token_counts = np.zeros(n_samples)
            print("Warning: No token counts, using all samples")
        
        # Ensure token_counts matches data length
        if len(token_counts) != n_samples:
            print(f"Warning: Token count mismatch ({len(token_counts)} vs {n_samples}), using all samples")
            token_counts = np.zeros(n_samples)
        
        # Filter
        filtered_indices = [i for i in range(n_samples) if token_counts[i] < max_tokens]
        n_filtered = len(filtered_indices)
        print(f"Filtering: {n_samples} -> {n_filtered} samples (removed {n_samples - n_filtered})")
        
        # Build stacked arrays from filtered samples
        activations_0pct = []
        activations_100pct = []
        activations_50pct = []
        labels = []
        
        for i in filtered_indices:
            d = data[i]
            activations_0pct.append(d['activations_0pct'])
            activations_100pct.append(d['activations_100pct'])
            if 'activations_50pct' in d:
                activations_50pct.append(d['activations_50pct'])
            labels.append(int(d.get('is_correct', d.get('label', 0))))
        
        filtered_data = {
            'activations_0pct': torch.stack(activations_0pct) if activations_0pct else None,
            'activations_100pct': torch.stack(activations_100pct) if activations_100pct else None,
            'labels': np.array(labels),
            'token_counts': token_counts[filtered_indices],
            'format': 'old_converted'
        }
        
        if activations_50pct:
            filtered_data['activations_50pct'] = torch.stack(activations_50pct)
        
        return filtered_data
    
    else:
        raise ValueError(f"Unknown data format: {type(data)}")


def train_success_probes(data, output_dir, test_size=0.2, random_state=42):
    """Train success probes on filtered data with proper train/test split."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract arrays
    X_0pct = data['activations_0pct']
    X_100pct = data['activations_100pct']
    X_50pct = data.get('activations_50pct')
    y = data['labels']
    
    # Convert to numpy
    if isinstance(X_0pct, torch.Tensor):
        X_0pct = X_0pct.numpy()
    if isinstance(X_100pct, torch.Tensor):
        X_100pct = X_100pct.numpy()
    if X_50pct is not None and isinstance(X_50pct, torch.Tensor):
        X_50pct = X_50pct.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    
    # Ensure y is int
    y = y.astype(int)
    
    n_samples = len(y)
    n_layers = X_0pct.shape[1]
    hidden_dim = X_0pct.shape[2]
    
    print(f"\n{'='*60}")
    print("TRAINING SUCCESS PROBES")
    print(f"{'='*60}")
    print(f"Samples: {n_samples}")
    print(f"Class balance: {y.sum()} correct, {n_samples - y.sum()} incorrect")
    print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Train/Test split: {1-test_size:.0%}/{test_size:.0%}")
    
    # Determine checkpoints
    checkpoints = {'0pct': X_0pct, '100pct': X_100pct}
    if X_50pct is not None:
        checkpoints['50pct'] = X_50pct
    
    # Results storage
    results = {cp: [] for cp in checkpoints}
    
    # Train probes for each checkpoint
    for cp_name in sorted(checkpoints.keys(), key=lambda x: int(x.replace('pct', ''))):
        X_all = checkpoints[cp_name]
        print(f"\n--- {cp_name} checkpoint ---")
        
        for layer in range(n_layers):
            X = X_all[:, layer, :]
            
            # Train/test split (same split for all layers for fair comparison)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError as e:
                # Fall back to non-stratified if class imbalance
                print(f"  Warning: Stratified split failed, using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            
            results[cp_name].append(acc)
            
            if layer % 5 == 0:
                print(f"  Layer {layer}: {acc:.3f}")
        
        best_idx = np.argmax(results[cp_name])
        best_acc = results[cp_name][best_idx]
        print(f"  Best: Layer {best_idx} = {best_acc:.3f}")
    
    # Generate 2D heatmap
    print(f"\n{'='*60}")
    print("GENERATING HEATMAP")
    print(f"{'='*60}")
    
    checkpoint_names = sorted(checkpoints.keys(), key=lambda x: int(x.replace('pct', '')))
    heatmap_data = np.zeros((n_layers, len(checkpoint_names)))
    
    for j, cp in enumerate(checkpoint_names):
        heatmap_data[:, j] = results[cp]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        heatmap_data,
        xticklabels=checkpoint_names,
        yticklabels=range(n_layers),
        cmap='RdYlGn',
        vmin=0.4,
        vmax=1.0,
        ax=ax,
        annot=False
    )
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Layer')
    ax.set_title(f'Success Probe Accuracy\n(n={n_samples}, filtered <1000 tokens)')
    
    plt.tight_layout()
    heatmap_path = f"{output_dir}/success_probe_heatmap_filtered.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved: {heatmap_path}")
    
    # Find overall best
    best_cp = None
    best_layer = None
    best_acc = 0
    for cp in checkpoint_names:
        for layer, acc in enumerate(results[cp]):
            if acc > best_acc:
                best_acc = acc
                best_cp = cp
                best_layer = layer
    
    print(f"\nOverall Best: Layer {best_layer}, {best_cp} = {best_acc:.3f}")
    
    # Save summary
    summary = {
        'n_samples': int(n_samples),
        'n_correct': int(y.sum()),
        'n_incorrect': int(n_samples - y.sum()),
        'n_layers': int(n_layers),
        'test_size': test_size,
        'random_state': random_state,
    }
    
    for cp in checkpoint_names:
        best_idx = np.argmax(results[cp])
        summary[f'{cp}_best_layer'] = int(best_idx)
        summary[f'{cp}_best_acc'] = float(results[cp][best_idx])
    
    summary['overall_best_checkpoint'] = best_cp
    summary['overall_best_layer'] = int(best_layer)
    summary['overall_best_acc'] = float(best_acc)
    
    summary_path = f"{output_dir}/summary_filtered.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    
    return results, summary


def analyze_token_distribution(data, output_dir):
    """Analyze token distribution in filtered data."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    token_counts = data.get('token_counts')
    labels = data['labels']
    
    if token_counts is None or len(token_counts) == 0:
        print("No token counts available for analysis")
        return
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Create DataFrame
    df = pd.DataFrame({
        'tokens': token_counts,
        'correct': labels.astype(int)
    })
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(df[df['correct']==1]['tokens'], bins=30, alpha=0.7, label='Correct', color='blue')
    ax.hist(df[df['correct']==0]['tokens'], bins=30, alpha=0.7, label='Incorrect', color='red')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Distribution (Filtered <1000)')
    ax.legend()
    
    # Box plot
    ax = axes[1]
    df['Correctness'] = df['correct'].map({1: 'Correct', 0: 'Incorrect'})
    df.boxplot(column='tokens', by='Correctness', ax=ax)
    ax.set_xlabel('Correctness')
    ax.set_ylabel('Token Count')
    ax.set_title('Token Distribution by Correctness')
    plt.suptitle('')
    
    plt.tight_layout()
    dist_path = f"{output_dir}/token_distribution_filtered.png"
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved: {dist_path}")
    
    # Print stats
    print("\nToken Statistics (Filtered):")
    correct_tokens = df[df['correct']==1]['tokens']
    incorrect_tokens = df[df['correct']==0]['tokens']
    print(f"  Correct: mean={correct_tokens.mean():.1f}, median={correct_tokens.median():.1f}, n={len(correct_tokens)}")
    print(f"  Incorrect: mean={incorrect_tokens.mean():.1f}, median={incorrect_tokens.median():.1f}, n={len(incorrect_tokens)}")


def main():
    parser = argparse.ArgumentParser(description='Train success probes with token filtering')
    parser.add_argument('--data_file', type=str, required=True, help='Path to probe_data.pt')
    parser.add_argument('--audit_file', type=str, default=None, help='Path to probe_audit.csv with num_tokens column')
    parser.add_argument('--output_dir', type=str, default='probe_results_filtered', help='Output directory')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Max tokens to include (default: 1000)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("="*60)
    print("FILTERED PROBE TRAINING")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Audit file: {args.audit_file}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.seed}")
    
    # Check files exist
    if not os.path.exists(args.data_file):
        print(f"\nERROR: Data file not found: {args.data_file}")
        print("Available .pt files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.pt'):
                print(f"  {f}")
        return
    
    # Load and filter
    data = load_and_filter_data(args.data_file, args.audit_file, args.max_tokens)
    
    if data['activations_0pct'] is None or len(data['labels']) == 0:
        print("ERROR: No data after filtering!")
        return
    
    # Train probes
    results, summary = train_success_probes(
        data, 
        args.output_dir, 
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Analyze tokens
    analyze_token_distribution(data, args.output_dir)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
