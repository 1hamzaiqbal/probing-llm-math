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
    data = torch.load(data_file, weights_only=False)
    
    # Check format (new vs old)
    if isinstance(data, dict) and 'activations_0pct' in data:
        # New format
        n_samples = len(data['labels'])
        print(f"Loaded {n_samples} samples (new format)")
        
        # Load audit file for token counts
        if audit_file and os.path.exists(audit_file):
            audit_df = pd.read_csv(audit_file)
            if 'num_tokens' in audit_df.columns:
                token_counts = audit_df['num_tokens'].values
            else:
                print("Warning: num_tokens not in audit file, using all samples")
                token_counts = np.zeros(n_samples)
        else:
            print("Warning: No audit file, using all samples")
            token_counts = np.zeros(n_samples)
        
        # Filter by token count
        mask = token_counts < max_tokens
        n_filtered = mask.sum()
        print(f"Filtering: {n_samples} -> {n_filtered} samples (removed {n_samples - n_filtered} with {max_tokens}+ tokens)")
        
        # Apply filter
        filtered_data = {
            'activations_0pct': data['activations_0pct'][mask],
            'activations_100pct': data['activations_100pct'][mask],
            'labels': data['labels'][mask] if isinstance(data['labels'], np.ndarray) else np.array(data['labels'])[mask],
        }
        
        # Handle 50pct if present
        if 'activations_50pct' in data and data['activations_50pct'] is not None:
            if isinstance(data['activations_50pct'], torch.Tensor):
                filtered_data['activations_50pct'] = data['activations_50pct'][mask]
            else:
                # It might be stored differently
                filtered_data['activations_50pct'] = None
        
        # Copy metadata
        filtered_data['metadata'] = data.get('metadata', {})
        filtered_data['topics'] = [data['topics'][i] for i in range(n_samples) if mask[i]] if 'topics' in data else None
        filtered_data['levels'] = [data['levels'][i] for i in range(n_samples) if mask[i]] if 'levels' in data else None
        filtered_data['token_counts'] = token_counts[mask]
        
        return filtered_data
        
    else:
        # Old format (list of dicts)
        print(f"Loaded {len(data)} samples (old format)")
        
        # Load audit file
        if audit_file and os.path.exists(audit_file):
            audit_df = pd.read_csv(audit_file)
            if 'num_tokens' in audit_df.columns:
                token_counts = audit_df['num_tokens'].values
            else:
                token_counts = np.zeros(len(data))
        else:
            token_counts = np.zeros(len(data))
        
        # Filter
        filtered_data = []
        filtered_tokens = []
        for i, d in enumerate(data):
            if token_counts[i] < max_tokens:
                filtered_data.append(d)
                filtered_tokens.append(token_counts[i])
        
        print(f"Filtering: {len(data)} -> {len(filtered_data)} samples")
        
        return filtered_data, np.array(filtered_tokens)


def train_success_probes(data, output_dir, test_size=0.2, random_state=42):
    """Train success probes on filtered data with proper train/test split."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data based on format
    if isinstance(data, dict):
        # New format
        X_0pct = data['activations_0pct']
        X_100pct = data['activations_100pct']
        X_50pct = data.get('activations_50pct')
        y = data['labels']
        
        if isinstance(X_0pct, torch.Tensor):
            X_0pct = X_0pct.numpy()
        if isinstance(X_100pct, torch.Tensor):
            X_100pct = X_100pct.numpy()
        if X_50pct is not None and isinstance(X_50pct, torch.Tensor):
            X_50pct = X_50pct.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
            
        n_layers = X_0pct.shape[1]
        hidden_dim = X_0pct.shape[2]
        
    else:
        # Old format
        X_0pct = np.stack([d['activations_0pct'].numpy() for d in data])
        X_100pct = np.stack([d['activations_100pct'].numpy() for d in data])
        X_50pct = np.stack([d['activations_50pct'].numpy() for d in data]) if 'activations_50pct' in data[0] else None
        y = np.array([d['is_correct'] for d in data]).astype(int)
        
        n_layers = X_0pct.shape[1]
        hidden_dim = X_0pct.shape[2]
    
    n_samples = len(y)
    print(f"\nTraining on {n_samples} samples")
    print(f"Class balance: {y.sum()} correct, {n_samples - y.sum()} incorrect")
    print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")
    
    # Determine checkpoints
    checkpoints = {'0pct': X_0pct, '100pct': X_100pct}
    if X_50pct is not None:
        checkpoints['50pct'] = X_50pct
    
    # Results storage
    results = {cp: {'logreg': [], 'layers': []} for cp in checkpoints}
    
    # Train probes for each checkpoint
    for cp_name, X_all in checkpoints.items():
        print(f"\n--- {cp_name} checkpoint ---")
        
        for layer in range(n_layers):
            X = X_all[:, layer, :]
            
            # Train/test split (IMPORTANT: same split for all layers)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            
            results[cp_name]['logreg'].append(acc)
            results[cp_name]['layers'].append(layer)
            
            if layer % 5 == 0:
                print(f"  Layer {layer}: {acc:.3f}")
        
        best_idx = np.argmax(results[cp_name]['logreg'])
        best_acc = results[cp_name]['logreg'][best_idx]
        print(f"  Best: Layer {best_idx} = {best_acc:.3f}")
    
    # Generate 2D heatmap
    print("\n" + "="*60)
    print("GENERATING HEATMAP")
    print("="*60)
    
    checkpoint_names = sorted(checkpoints.keys(), key=lambda x: int(x.replace('pct', '')))
    heatmap_data = np.zeros((n_layers, len(checkpoint_names)))
    
    for j, cp in enumerate(checkpoint_names):
        heatmap_data[:, j] = results[cp]['logreg']
    
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
    ax.set_title(f'Success Probe Accuracy (n={n_samples}, filtered <{1000} tokens)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_probe_heatmap_filtered.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/success_probe_heatmap_filtered.png")
    
    # Find overall best
    best_cp = None
    best_layer = None
    best_acc = 0
    for cp in checkpoint_names:
        for layer, acc in enumerate(results[cp]['logreg']):
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
        'n_layers': n_layers,
        'test_size': test_size,
        'random_state': random_state,
    }
    
    for cp in checkpoint_names:
        best_idx = np.argmax(results[cp]['logreg'])
        summary[f'{cp}_best_layer'] = int(best_idx)
        summary[f'{cp}_best_acc'] = float(results[cp]['logreg'][best_idx])
    
    summary['overall_best_checkpoint'] = best_cp
    summary['overall_best_layer'] = int(best_layer)
    summary['overall_best_acc'] = float(best_acc)
    
    with open(f"{output_dir}/summary_filtered.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir}/summary_filtered.json")
    
    return results, summary


def analyze_token_distribution(data, output_dir):
    """Analyze token distribution in filtered data."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(data, dict):
        token_counts = data.get('token_counts', None)
        labels = data['labels']
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
    else:
        return  # Can't analyze without token counts
    
    if token_counts is None:
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'tokens': token_counts,
        'correct': labels
    })
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(df[df['correct']==1]['tokens'], bins=30, alpha=0.7, label='Correct', color='blue')
    ax.hist(df[df['correct']==0]['tokens'], bins=30, alpha=0.7, label='Incorrect', color='red')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Distribution (Filtered)')
    ax.legend()
    ax.axvline(x=1000, color='black', linestyle='--', label='Filter threshold')
    
    # Box plot
    ax = axes[1]
    df['Correctness'] = df['correct'].map({1: 'Correct', 0: 'Incorrect'})
    df.boxplot(column='tokens', by='Correctness', ax=ax)
    ax.set_xlabel('Correctness')
    ax.set_ylabel('Token Count')
    ax.set_title('Token Distribution by Correctness (Filtered)')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_distribution_filtered.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/token_distribution_filtered.png")
    
    # Print stats
    print("\nToken Statistics (Filtered):")
    print(f"  Correct: mean={df[df['correct']==1]['tokens'].mean():.1f}, median={df[df['correct']==1]['tokens'].median():.1f}")
    print(f"  Incorrect: mean={df[df['correct']==0]['tokens'].mean():.1f}, median={df[df['correct']==0]['tokens'].median():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to probe_data.pt')
    parser.add_argument('--audit_file', type=str, default=None, help='Path to probe_audit.csv')
    parser.add_argument('--output_dir', type=str, default='probe_results_filtered', help='Output directory')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Max tokens to include')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
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
    
    # Load and filter
    data = load_and_filter_data(args.data_file, args.audit_file, args.max_tokens)
    
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
    print("SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()

