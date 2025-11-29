"""
Train linear probes on collected hidden state data.

Supports three probe types:
1. Success Probe: Predict if answer will be correct (binary)
2. Topic Probe: Predict problem topic (multi-class)
3. Difficulty Probe: Predict problem difficulty level (regression or multi-class)
"""

import torch
import numpy as np
import argparse
import json
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data(data_file):
    """Load probe data and extract metadata."""
    print(f"Loading data from {data_file}...")
    data = torch.load(data_file)
    
    print(f"Loaded {len(data)} samples")
    
    # Extract metadata
    num_layers = data[0]['activations_0pct'].shape[0]
    hidden_dim = data[0]['activations_0pct'].shape[1]
    
    has_50pct = 'activations_50pct' in data[0]
    
    # Get unique topics and levels
    topics = list(set(d['topic'] for d in data))
    levels = list(set(d['level_num'] for d in data))
    
    print(f"Num layers: {num_layers}, Hidden dim: {hidden_dim}")
    print(f"Topics: {topics}")
    print(f"Levels: {sorted(levels)}")
    print(f"Has 50% checkpoints: {has_50pct}")
    
    return data, num_layers, hidden_dim, has_50pct


def train_probe_per_layer(X_by_layer, y, num_layers, probe_type='classification', 
                          test_size=0.2, random_state=42):
    """
    Train a probe for each layer and return accuracy/metrics per layer.
    
    Args:
        X_by_layer: Function that takes layer_idx and returns feature matrix
        y: Labels
        num_layers: Number of layers
        probe_type: 'classification' or 'regression'
        
    Returns:
        List of accuracy/R^2 scores per layer
    """
    results = []
    best_layer = -1
    best_score = -1
    
    # Check if we have enough samples per class for stratified split
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    
    # Need at least 2 samples per class for stratified split
    use_stratify = probe_type == 'classification' and min_count >= 2
    if probe_type == 'classification' and min_count < 2:
        print(f"  WARNING: Class with only {min_count} sample(s), using non-stratified split")
    
    for layer_idx in range(num_layers):
        X = X_by_layer(layer_idx)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if use_stratify else None
            )
        except ValueError as e:
            # Fallback to non-stratified if stratified fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Check if test set has enough variety
        if len(np.unique(y_train)) < 2:
            # Not enough classes in training set
            score = 0.0
        elif probe_type == 'classification':
            clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        else:
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)  # R^2
        
        results.append(score)
        
        if score > best_score:
            best_score = score
            best_layer = layer_idx
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}: {score:.3f}")
    
    return results, best_layer, best_score


def train_success_probe(data, num_layers, output_dir, checkpoints=['0pct', '50pct', '100pct']):
    """
    Train success prediction probes (binary: correct vs incorrect).
    """
    print("\n" + "="*60)
    print("TRAINING SUCCESS PROBES")
    print("="*60)
    
    y = np.array([1 if d['is_correct'] else 0 for d in data])
    print(f"Class balance: {sum(y)} correct, {len(y)-sum(y)} incorrect")
    
    if len(set(y)) < 2:
        print("ERROR: Need both correct and incorrect samples!")
        return {}
    
    all_results = {}
    
    for checkpoint in checkpoints:
        key = f'activations_{checkpoint}'
        
        # Check if this checkpoint exists
        if key not in data[0]:
            print(f"  Skipping {checkpoint} (not in data)")
            continue
        
        print(f"\n--- {checkpoint} checkpoint ---")
        
        def get_X(layer_idx):
            return np.array([d[key][layer_idx].numpy() for d in data])
        
        results, best_layer, best_score = train_probe_per_layer(
            get_X, y, num_layers, probe_type='classification'
        )
        
        all_results[checkpoint] = {
            'results': results,
            'best_layer': best_layer,
            'best_score': best_score,
        }
        
        print(f"  Best: Layer {best_layer} with accuracy {best_score:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    for checkpoint, res in all_results.items():
        plt.plot(res['results'], label=f"{checkpoint} (best={res['best_score']:.3f} @ L{res['best_layer']})")
    
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.title("Success Probe Accuracy by Layer and Checkpoint")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_probe_accuracy.png", dpi=150)
    print(f"\nSaved: {output_dir}/success_probe_accuracy.png")
    
    return all_results


def train_topic_probe(data, num_layers, output_dir):
    """
    Train topic classification probes (multi-class).
    """
    print("\n" + "="*60)
    print("TRAINING TOPIC PROBES")
    print("="*60)
    
    # Encode topics
    topics = [d['topic'] for d in data]
    le = LabelEncoder()
    y = le.fit_transform(topics)
    
    print(f"Topics: {list(le.classes_)}")
    print(f"Distribution: {dict(zip(le.classes_, np.bincount(y)))}")
    
    if len(le.classes_) < 2:
        print("ERROR: Need at least 2 topics!")
        return {}
    
    # Check minimum samples per class
    min_samples = min(np.bincount(y))
    if min_samples < 2:
        print(f"WARNING: Topic '{le.classes_[np.argmin(np.bincount(y))]}' has only {min_samples} sample(s)")
        print("         Results may be unreliable. Collect more data for robust probing.")
    
    # Use 0% checkpoint (question-only) for topic probing
    print("\n--- 0pct checkpoint (question-only) ---")
    
    def get_X(layer_idx):
        return np.array([d['activations_0pct'][layer_idx].numpy() for d in data])
    
    results, best_layer, best_score = train_probe_per_layer(
        get_X, y, num_layers, probe_type='classification'
    )
    
    print(f"Best: Layer {best_layer} with accuracy {best_score:.3f}")
    
    # Detailed evaluation at best layer
    X_best = get_X(best_layer)
    X_train, X_test, y_train, y_test = train_test_split(
        X_best, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\nClassification Report (Layer {best_layer}):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results, 'b-', linewidth=2)
    plt.axhline(y=1.0/len(le.classes_), color='gray', linestyle='--', alpha=0.5, label=f'Chance ({1.0/len(le.classes_):.2f})')
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.title(f"Topic Probe Accuracy by Layer (Best: {best_score:.3f} @ Layer {best_layer})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topic_probe_accuracy.png", dpi=150)
    print(f"\nSaved: {output_dir}/topic_probe_accuracy.png")
    
    return {
        '0pct': {
            'results': results,
            'best_layer': best_layer,
            'best_score': best_score,
            'classes': list(le.classes_),
        }
    }


def train_difficulty_probe(data, num_layers, output_dir, mode='classification'):
    """
    Train difficulty prediction probes.
    
    Args:
        mode: 'classification' (5-class) or 'regression' (continuous 1-5)
    """
    print("\n" + "="*60)
    print(f"TRAINING DIFFICULTY PROBES ({mode})")
    print("="*60)
    
    y = np.array([d['level_num'] for d in data])
    
    # Filter out unknown levels (0)
    valid_mask = y > 0
    if not all(valid_mask):
        print(f"Filtering out {sum(~valid_mask)} samples with unknown difficulty")
        data_filtered = [d for d, m in zip(data, valid_mask) if m]
        y = y[valid_mask]
    else:
        data_filtered = data
    
    print(f"Level distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if len(set(y)) < 2:
        print("ERROR: Need at least 2 difficulty levels!")
        return {}
    
    # Use 0% checkpoint (question-only) for difficulty probing
    print("\n--- 0pct checkpoint (question-only) ---")
    
    def get_X(layer_idx):
        return np.array([d['activations_0pct'][layer_idx].numpy() for d in data_filtered])
    
    results, best_layer, best_score = train_probe_per_layer(
        get_X, y, num_layers, probe_type=mode
    )
    
    metric_name = "Accuracy" if mode == 'classification' else "R²"
    print(f"Best: Layer {best_layer} with {metric_name} {best_score:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results, 'g-', linewidth=2)
    if mode == 'classification':
        plt.axhline(y=1.0/len(set(y)), color='gray', linestyle='--', alpha=0.5, label='Chance')
    else:
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel("Layer Index")
    plt.ylabel(metric_name)
    plt.title(f"Difficulty Probe {metric_name} by Layer (Best: {best_score:.3f} @ Layer {best_layer})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/difficulty_probe_{mode}.png", dpi=150)
    print(f"\nSaved: {output_dir}/difficulty_probe_{mode}.png")
    
    return {
        '0pct': {
            'results': results,
            'best_layer': best_layer,
            'best_score': best_score,
            'mode': mode,
        }
    }


def generate_pca_visualization(data, num_layers, output_dir, color_by='correctness'):
    """
    Generate PCA visualization of hidden states.
    """
    print("\n" + "="*60)
    print(f"GENERATING PCA VISUALIZATION (colored by {color_by})")
    print("="*60)
    
    # Find a good layer (middle-ish)
    target_layer = min(20, num_layers - 1)
    
    X = np.array([d['activations_0pct'][target_layer].numpy() for d in data])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    if color_by == 'correctness':
        colors = ['red' if not d['is_correct'] else 'blue' for d in data]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
        plt.title(f"PCA of Hidden States (Layer {target_layer})\nRed: Incorrect, Blue: Correct")
    
    elif color_by == 'topic':
        topics = [d['topic'] for d in data]
        unique_topics = list(set(topics))
        cmap = plt.cm.get_cmap('tab10', len(unique_topics))
        colors = [cmap(unique_topics.index(t)) for t in topics]
        
        for i, topic in enumerate(unique_topics):
            mask = [t == topic for t in topics]
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[cmap(i)], 
                       label=topic, alpha=0.6, s=50)
        plt.legend()
        plt.title(f"PCA of Hidden States (Layer {target_layer}) - By Topic")
    
    elif color_by == 'difficulty':
        levels = np.array([d['level_num'] for d in data])
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=levels, 
                             cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Difficulty Level')
        plt.title(f"PCA of Hidden States (Layer {target_layer}) - By Difficulty")
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_layer{target_layer}_{color_by}.png", dpi=150)
    print(f"Saved: {output_dir}/pca_layer{target_layer}_{color_by}.png")


def train_all_probes(data_file="probe_data.pt", output_dir="probe_results"):
    """Main function to train all probe types."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data, num_layers, hidden_dim, has_50pct = load_data(data_file)
    
    # Determine which checkpoints to use
    checkpoints = ['0pct', '100pct']
    if has_50pct:
        checkpoints.insert(1, '50pct')
    
    all_results = {}
    
    # 1. Success Probes
    all_results['success'] = train_success_probe(data, num_layers, output_dir, checkpoints)
    
    # 2. Topic Probes
    all_results['topic'] = train_topic_probe(data, num_layers, output_dir)
    
    # 3. Difficulty Probes (both classification and regression)
    all_results['difficulty_class'] = train_difficulty_probe(
        data, num_layers, output_dir, mode='classification'
    )
    all_results['difficulty_regress'] = train_difficulty_probe(
        data, num_layers, output_dir, mode='regression'
    )
    
    # 4. PCA Visualizations
    for color_by in ['correctness', 'topic', 'difficulty']:
        generate_pca_visualization(data, num_layers, output_dir, color_by=color_by)
    
    # Save summary
    summary = {
        'num_samples': len(data),
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'has_50pct': has_50pct,
    }
    
    # Add best results
    for probe_type, results in all_results.items():
        if results:
            for checkpoint, res in results.items():
                if isinstance(res, dict) and 'best_score' in res:
                    summary[f'{probe_type}_{checkpoint}_best_layer'] = res['best_layer']
                    summary[f'{probe_type}_{checkpoint}_best_score'] = res['best_score']
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        # Convert numpy types for JSON serialization
        summary_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in summary.items()}
        json.dump(summary_json, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear probes on hidden state data.")
    parser.add_argument("--data_file", type=str, default="probe_data.pt",
                        help="Input data file from collect_probe_data.py")
    parser.add_argument("--output_dir", type=str, default="probe_results",
                        help="Output directory for results and plots")
    
    args = parser.parse_args()
    
    train_all_probes(
        data_file=args.data_file,
        output_dir=args.output_dir,
    )
