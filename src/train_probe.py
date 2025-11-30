"""
Train linear probes on collected hidden state data.

Supports three probe types:
1. Success Probe: Predict if answer will be correct (binary)
2. Topic Probe: Predict problem topic (multi-class)
3. Difficulty Probe: Predict problem difficulty level (regression or multi-class)

Enhanced with:
- Difference-of-means probes (Geometry of Truth style)
- 2D heatmap visualization (layer × checkpoint)
- Control task (shuffled labels) for validation
- MLP probes (non-linear)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import json
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

        
def load_data(data_file):
    """Load probe data and extract metadata. Handles both old and new formats."""
    print(f"Loading data from {data_file}...")
    loaded = torch.load(data_file)
    
    # Handle new format with metadata
    if isinstance(loaded, dict) and 'data' in loaded and 'metadata' in loaded:
        data = loaded['data']
        metadata = loaded['metadata']
        print(f"Loaded {len(data)} samples (new format with metadata)")
        print(f"Metadata: {metadata}")
    else:
        # Old format: just a list of samples
        data = loaded
        metadata = None
        print(f"Loaded {len(data)} samples (legacy format)")
    
    # Extract dimensions from data
    num_layers = data[0]['activations_0pct'].shape[0]
    hidden_dim = data[0]['activations_0pct'].shape[1]
    
    has_50pct = 'activations_50pct' in data[0]
    has_mean_pooling = 'activations_0pct_mean' in data[0]
    
    # Get unique topics and levels
    topics = list(set(d['topic'] for d in data))
    levels = list(set(d['level_num'] for d in data))
    
    print(f"Num layers: {num_layers}, Hidden dim: {hidden_dim}")
    print(f"Topics: {topics}")
    print(f"Levels: {sorted(levels)}")
    print(f"Has 50% checkpoints: {has_50pct}")
    print(f"Has mean pooling: {has_mean_pooling}")
    
    return data, num_layers, hidden_dim, has_50pct, has_mean_pooling


# =============================================================================
# DIFFERENCE-OF-MEANS PROBE (from Geometry of Truth)
# =============================================================================

def train_diff_of_means_probe(X_train, y_train, X_test, y_test):
    """
    Train a difference-of-means probe (Marks & Tegmark style).
    
    Direction: w = μ_positive - μ_negative
    Decision: sign(w · x + b) where b = -(μ_pos + μ_neg) · w / 2
    
    Returns accuracy on test set.
    """
    # Compute class means
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    
    if sum(pos_mask) == 0 or sum(neg_mask) == 0:
        return 0.5  # Can't compute means
    
    mu_pos = X_train[pos_mask].mean(axis=0)
    mu_neg = X_train[neg_mask].mean(axis=0)
    
    # Direction vector
    w = mu_pos - mu_neg
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return 0.5
    w = w / w_norm  # Normalize
    
    # Bias (decision boundary at midpoint)
    b = -0.5 * (mu_pos + mu_neg) @ w
    
    # Predict on test
    scores = X_test @ w + b
    y_pred = (scores > 0).astype(int)
    
    return accuracy_score(y_test, y_pred)


# =============================================================================
# PROBE TRAINING WITH MULTIPLE METHODS
# =============================================================================

def train_probe_per_layer(X_by_layer, y, num_layers, probe_type='classification', 
                          test_size=0.2, random_state=42, include_diff_of_means=False):
    """
    Train a probe for each layer and return accuracy/metrics per layer.
    
    Args:
        X_by_layer: Function that takes layer_idx and returns feature matrix
        y: Labels
        num_layers: Number of layers
        probe_type: 'classification' or 'regression'
        include_diff_of_means: If True, also train diff-of-means probes (binary only)
        
    Returns:
        results: List of logistic regression scores per layer
        best_layer: Best performing layer
        best_score: Best score achieved
        diff_results: (optional) List of diff-of-means scores if requested
    """
    results = []
    diff_results = [] if include_diff_of_means else None
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
        except ValueError:
            # Fallback to non-stratified if stratified fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Check if test set has enough variety
        if len(np.unique(y_train)) < 2:
            score = 0.0
            diff_score = 0.5
        elif probe_type == 'classification':
            clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            # Diff-of-means (only for binary)
            if include_diff_of_means and len(unique) == 2:
                diff_score = train_diff_of_means_probe(X_train, y_train, X_test, y_test)
        else:
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)  # R^2
            diff_score = 0.0
        
        results.append(score)
        if include_diff_of_means:
            diff_results.append(diff_score)
        
        if score > best_score:
            best_score = score
            best_layer = layer_idx
        
        if layer_idx % 5 == 0:
            if include_diff_of_means:
                print(f"  Layer {layer_idx}: LogReg={score:.3f}, DiffMeans={diff_score:.3f}")
            else:
                print(f"  Layer {layer_idx}: {score:.3f}")
    
    if include_diff_of_means:
        return results, best_layer, best_score, diff_results
    return results, best_layer, best_score


# =============================================================================
# CONTROL TASK (SHUFFLED LABELS)
# =============================================================================

def train_control_probe(data, num_layers, output_dir, checkpoint='0pct'):
    """
    Train probes with shuffled labels to verify we're not just overfitting.
    
    Expected result: ~50% accuracy (chance) for binary classification.
    """
    print("\n" + "="*60)
    print("CONTROL TASK (SHUFFLED LABELS)")
    print("="*60)
    
    y_real = np.array([1 if d['is_correct'] else 0 for d in data])
    
    # Shuffle labels
    np.random.seed(42)
    y_shuffled = np.random.permutation(y_real)
    
    key = f'activations_{checkpoint}'
    
    def get_X(layer_idx):
        return np.array([d[key][layer_idx].numpy() for d in data])
    
    print(f"Training on shuffled labels (should get ~50% accuracy)...")
    
    results, best_layer, best_score = train_probe_per_layer(
        get_X, y_shuffled, num_layers, probe_type='classification'
    )
    
    mean_acc = np.mean(results)
    print(f"\nControl task results:")
    print(f"  Mean accuracy: {mean_acc:.3f} (expected: ~0.50)")
    print(f"  Best layer: {best_layer} with {best_score:.3f}")
    
    if mean_acc > 0.65:
        print("  ⚠️  WARNING: Control accuracy suspiciously high! May indicate overfitting.")
    else:
        print("  ✓ Control task passed (close to chance)")
    
    # Plot comparison with real labels
    plt.figure(figsize=(10, 6))
    plt.plot(results, 'r-', alpha=0.7, label=f'Shuffled (mean={mean_acc:.2f})')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.title("Control Task: Probe Accuracy with Shuffled Labels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_task_shuffled.png", dpi=150)
    print(f"Saved: {output_dir}/control_task_shuffled.png")
    
    return {'results': results, 'mean': mean_acc, 'best_layer': best_layer, 'best_score': best_score}


# =============================================================================
# 2D HEATMAP (LAYER × CHECKPOINT)
# =============================================================================

def generate_2d_heatmap(data, num_layers, output_dir, checkpoints=['0pct', '50pct', '100pct']):
    """
    Generate 2D heatmap of success probe accuracy (layer × checkpoint).
    
    Shows how predictive signal evolves through layers and generation progress.
    """
    print("\n" + "="*60)
    print("GENERATING 2D HEATMAP (Layer × Checkpoint)")
    print("="*60)
    
    y = np.array([1 if d['is_correct'] else 0 for d in data])
    
    if len(set(y)) < 2:
        print("ERROR: Need both correct and incorrect samples!")
        return None
    
    # Filter to available checkpoints
    available = [cp for cp in checkpoints if f'activations_{cp}' in data[0]]
    if len(available) < 2:
        print("Need at least 2 checkpoints for heatmap")
        return None
    
    print(f"Checkpoints: {available}")
    
    # Build accuracy matrix
    accuracy_matrix = np.zeros((num_layers, len(available)))
    diff_means_matrix = np.zeros((num_layers, len(available)))
    
    for j, checkpoint in enumerate(available):
        key = f'activations_{checkpoint}'
        
        def get_X(layer_idx):
            return np.array([d[key][layer_idx].numpy() for d in data])
        
        print(f"  Processing {checkpoint}...")
        
        for layer_idx in range(num_layers):
            X = get_X(layer_idx)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Logistic regression
            if len(np.unique(y_train)) >= 2:
                clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
                clf.fit(X_train, y_train)
                accuracy_matrix[layer_idx, j] = accuracy_score(y_test, clf.predict(X_test))
                
                # Diff-of-means
                diff_means_matrix[layer_idx, j] = train_diff_of_means_probe(
                    X_train, y_train, X_test, y_test
                )
            else:
                accuracy_matrix[layer_idx, j] = 0.5
                diff_means_matrix[layer_idx, j] = 0.5
    
    # Plot logistic regression heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Logistic Regression
    ax1 = axes[0]
    sns.heatmap(
        accuracy_matrix,
        cmap='RdYlGn',
        vmin=0.4,
        vmax=1.0,
        xticklabels=available,
        yticklabels=range(num_layers),
        ax=ax1,
        cbar_kws={'label': 'Accuracy'}
    )
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Layer")
    ax1.set_title("Logistic Regression Probe")
    
    # Only show every 5th layer label
    ax1.set_yticks(range(0, num_layers, 5))
    ax1.set_yticklabels(range(0, num_layers, 5))
    
    # Diff-of-Means
    ax2 = axes[1]
    sns.heatmap(
        diff_means_matrix,
        cmap='RdYlGn',
        vmin=0.4,
        vmax=1.0,
        xticklabels=available,
        yticklabels=range(num_layers),
        ax=ax2,
        cbar_kws={'label': 'Accuracy'}
    )
    ax2.set_xlabel("Checkpoint")
    ax2.set_ylabel("Layer")
    ax2.set_title("Difference-of-Means Probe")
    ax2.set_yticks(range(0, num_layers, 5))
    ax2.set_yticklabels(range(0, num_layers, 5))
    
    plt.suptitle("Success Probe Accuracy: Layer × Checkpoint", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_probe_heatmap_2d.png", dpi=150)
    print(f"Saved: {output_dir}/success_probe_heatmap_2d.png")
    
    # Find best (layer, checkpoint) for each method
    best_logreg = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_diff = np.unravel_index(np.argmax(diff_means_matrix), diff_means_matrix.shape)
    
    print(f"\nBest LogReg: Layer {best_logreg[0]}, {available[best_logreg[1]]} = {accuracy_matrix[best_logreg]:.3f}")
    print(f"Best DiffMeans: Layer {best_diff[0]}, {available[best_diff[1]]} = {diff_means_matrix[best_diff]:.3f}")
    
    return {
        'accuracy_matrix': accuracy_matrix,
        'diff_means_matrix': diff_means_matrix,
        'checkpoints': available,
    }


# =============================================================================
# SUCCESS PROBE TRAINING
# =============================================================================

def train_success_probe(data, num_layers, output_dir, checkpoints=['0pct', '50pct', '100pct']):
    """
    Train success prediction probes (binary: correct vs incorrect).
    Now includes both logistic regression and diff-of-means.
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
        
        if key not in data[0]:
            print(f"  Skipping {checkpoint} (not in data)")
            continue
        
        print(f"\n--- {checkpoint} checkpoint ---")
        
        def get_X(layer_idx):
            return np.array([d[key][layer_idx].numpy() for d in data])
        
        # Train both logistic regression and diff-of-means
        results, best_layer, best_score, diff_results = train_probe_per_layer(
            get_X, y, num_layers, probe_type='classification', include_diff_of_means=True
        )
        
        # Find best diff-of-means layer
        diff_best_layer = np.argmax(diff_results)
        diff_best_score = diff_results[diff_best_layer]
        
        all_results[checkpoint] = {
            'logreg_results': results,
            'logreg_best_layer': best_layer,
            'logreg_best_score': best_score,
            'diff_results': diff_results,
            'diff_best_layer': diff_best_layer,
            'diff_best_score': diff_best_score,
        }
        
        print(f"  LogReg Best: Layer {best_layer} = {best_score:.3f}")
        print(f"  DiffMeans Best: Layer {diff_best_layer} = {diff_best_score:.3f}")
    
    # Plot comparison (logistic regression curves)
    plt.figure(figsize=(12, 6))
    for checkpoint, res in all_results.items():
        plt.plot(res['logreg_results'], 
                label=f"{checkpoint} (LogReg, best={res['logreg_best_score']:.2f} @ L{res['logreg_best_layer']})")
        plt.plot(res['diff_results'], '--', alpha=0.6,
                label=f"{checkpoint} (DiffMeans, best={res['diff_best_score']:.2f} @ L{res['diff_best_layer']})")
    
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.title("Success Probe: Logistic Regression vs Difference-of-Means")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_probe_accuracy.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/success_probe_accuracy.png")
    
    return all_results


# =============================================================================
# TOPIC PROBE
# =============================================================================

def train_topic_probe(data, num_layers, output_dir):
    """Train topic classification probes (multi-class)."""
    print("\n" + "="*60)
    print("TRAINING TOPIC PROBES")
    print("="*60)
    
    topics = [d['topic'] for d in data]
    le = LabelEncoder()
    y = le.fit_transform(topics)
    
    print(f"Topics: {list(le.classes_)}")
    print(f"Distribution: {dict(zip(le.classes_, np.bincount(y)))}")
    
    if len(le.classes_) < 2:
        print("ERROR: Need at least 2 topics!")
        return {}
    
    min_samples = min(np.bincount(y))
    if min_samples < 2:
        print(f"WARNING: Topic '{le.classes_[np.argmin(np.bincount(y))]}' has only {min_samples} sample(s)")
        print("         Results may be unreliable. Collect more data for robust probing.")
    
    print("\n--- 0pct checkpoint (question-only) ---")
    
    def get_X(layer_idx):
        return np.array([d['activations_0pct'][layer_idx].numpy() for d in data])
    
    results, best_layer, best_score = train_probe_per_layer(
        get_X, y, num_layers, probe_type='classification'
    )
    
    print(f"Best: Layer {best_layer} with accuracy {best_score:.3f}")
    
    # Detailed evaluation at best layer (if enough samples)
    if min_samples >= 2:
        X_best = get_X(best_layer)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_best, y, test_size=0.2, random_state=42, stratify=y
            )
            clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            print(f"\nClassification Report (Layer {best_layer}):")
            print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results, 'b-', linewidth=2)
    plt.axhline(y=1.0/len(le.classes_), color='gray', linestyle='--', alpha=0.5, 
                label=f'Chance ({1.0/len(le.classes_):.2f})')
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


# =============================================================================
# DIFFICULTY PROBE
# =============================================================================

def train_difficulty_probe(data, num_layers, output_dir, mode='classification'):
    """Train difficulty prediction probes."""
    print("\n" + "="*60)
    print(f"TRAINING DIFFICULTY PROBES ({mode})")
    print("="*60)
    
    y = np.array([d['level_num'] for d in data])
    
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


# =============================================================================
# PCA VISUALIZATION
# =============================================================================

def generate_pca_visualization(data, num_layers, output_dir, color_by='correctness'):
    """Generate PCA visualization of hidden states."""
    print("\n" + "="*60)
    print(f"GENERATING PCA VISUALIZATION (colored by {color_by})")
    print("="*60)
    
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
        
        for i, topic in enumerate(unique_topics):
            mask = np.array([t == topic for t in topics])
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


# =============================================================================
# MLP PROBES (NON-LINEAR)
# =============================================================================

class MLPProbe(nn.Module):
    """Simple 2-layer MLP for probing."""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_mlp_probe(X_train, y_train, X_test, y_test, input_dim, num_classes=2, 
                    epochs=100, lr=0.001, batch_size=32, verbose=False):
    """
    Train an MLP probe and return test accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = MLPProbe(input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean().item()
    
    return accuracy


def train_mlp_success_probe(data, num_layers, output_dir, checkpoints=['0pct', '50pct', '100pct']):
    """
    Train MLP probes for success prediction and compare with linear probes.
    """
    print("\n" + "="*60)
    print("TRAINING MLP SUCCESS PROBES")
    print("="*60)
    
    y = np.array([1 if d['is_correct'] else 0 for d in data])
    hidden_dim = data[0]['activations_0pct'].shape[1]
    
    results = {}
    
    for checkpoint in checkpoints:
        key = f'activations_{checkpoint}'
        if key not in data[0]:
            continue
            
        print(f"\n--- {checkpoint} checkpoint ---")
        
        mlp_results = []
        linear_results = []
        
        for layer_idx in range(num_layers):
            X = np.array([d[key][layer_idx].numpy() for d in data])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # MLP probe
            mlp_acc = train_mlp_probe(X_train, y_train, X_test, y_test, hidden_dim, num_classes=2)
            mlp_results.append(mlp_acc)
            
            # Linear probe for comparison
            clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
            clf.fit(X_train, y_train)
            linear_acc = accuracy_score(y_test, clf.predict(X_test))
            linear_results.append(linear_acc)
            
            if layer_idx % 5 == 0:
                print(f"  Layer {layer_idx}: Linear={linear_acc:.3f}, MLP={mlp_acc:.3f}")
        
        best_mlp_layer = np.argmax(mlp_results)
        best_linear_layer = np.argmax(linear_results)
        
        print(f"\n  Linear Best: Layer {best_linear_layer} = {linear_results[best_linear_layer]:.3f}")
        print(f"  MLP Best: Layer {best_mlp_layer} = {mlp_results[best_mlp_layer]:.3f}")
        
        results[checkpoint] = {
            'linear': linear_results,
            'mlp': mlp_results,
            'linear_best': (best_linear_layer, linear_results[best_linear_layer]),
            'mlp_best': (best_mlp_layer, mlp_results[best_mlp_layer])
        }
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (checkpoint, res) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(res['linear'], 'b-', alpha=0.7, label='Linear (LogReg)', linewidth=2)
        ax.plot(res['mlp'], 'r-', alpha=0.7, label='MLP', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{checkpoint}: Linear vs MLP")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mlp_vs_linear.png", dpi=150)
    print(f"\nSaved: {output_dir}/mlp_vs_linear.png")
    
    # Summary
    print("\n" + "-"*60)
    print("MLP vs LINEAR SUMMARY")
    print("-"*60)
    print(f"{'Checkpoint':<12} {'Linear Best':<15} {'MLP Best':<15} {'Δ':<10}")
    print("-"*60)
    
    for checkpoint, res in results.items():
        linear_best = res['linear_best'][1]
        mlp_best = res['mlp_best'][1]
        delta = mlp_best - linear_best
        symbol = "+" if delta > 0 else ""
        print(f"{checkpoint:<12} {linear_best:<15.3f} {mlp_best:<15.3f} {symbol}{delta:<10.3f}")
    
    return results


# =============================================================================
# MEAN POOLING vs LAST TOKEN COMPARISON
# =============================================================================

def compare_pooling_methods(data, num_layers, output_dir, checkpoints=['0pct', '100pct']):
    """
    Compare mean pooling vs last token for success prediction.
    
    This helps determine which pooling method captures more useful signal.
    """
    print("\n" + "="*60)
    print("COMPARING POOLING METHODS (Mean vs Last Token)")
    print("="*60)
    
    # Check if mean pooling data is available
    if 'activations_0pct_mean' not in data[0]:
        print("Mean pooling data not available. Skipping comparison.")
        return None
    
    y = np.array([1 if d['is_correct'] else 0 for d in data])
    
    results = {
        'last_token': {},
        'mean_pool': {}
    }
    
    # Filter to available checkpoints
    available_last = [cp for cp in checkpoints if f'activations_{cp}' in data[0]]
    available_mean = [cp for cp in checkpoints if f'activations_{cp}_mean' in data[0]]
    
    for checkpoint in available_last:
        if checkpoint not in available_mean:
            continue
            
        print(f"\n--- {checkpoint} checkpoint ---")
        
        # Last token
        key_last = f'activations_{checkpoint}'
        def get_X_last(layer_idx):
            return np.array([d[key_last][layer_idx].numpy() for d in data])
        
        res_last, best_layer_last, best_score_last, diff_last = train_probe_per_layer(
            get_X_last, y, num_layers, probe_type='classification', include_diff_of_means=True
        )
        
        # Mean pool
        key_mean = f'activations_{checkpoint}_mean'
        def get_X_mean(layer_idx):
            return np.array([d[key_mean][layer_idx].numpy() for d in data])
        
        res_mean, best_layer_mean, best_score_mean, diff_mean = train_probe_per_layer(
            get_X_mean, y, num_layers, probe_type='classification', include_diff_of_means=True
        )
        
        results['last_token'][checkpoint] = {
            'results': res_last, 'best_layer': best_layer_last, 'best_score': best_score_last,
            'diff_results': diff_last
        }
        results['mean_pool'][checkpoint] = {
            'results': res_mean, 'best_layer': best_layer_mean, 'best_score': best_score_mean,
            'diff_results': diff_mean
        }
        
        print(f"\n  Last Token: Best = {best_score_last:.3f} @ Layer {best_layer_last}")
        print(f"  Mean Pool:  Best = {best_score_mean:.3f} @ Layer {best_layer_mean}")
        
        diff = best_score_mean - best_score_last
        if diff > 0.02:
            print(f"  → Mean pooling is better (+{diff:.3f})")
        elif diff < -0.02:
            print(f"  → Last token is better (+{-diff:.3f})")
        else:
            print(f"  → Similar performance (Δ={diff:.3f})")
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(available_last), figsize=(5*len(available_last), 5))
    if len(available_last) == 1:
        axes = [axes]
    
    for idx, checkpoint in enumerate(available_last):
        if checkpoint not in available_mean:
            continue
            
        ax = axes[idx]
        
        res_last = results['last_token'][checkpoint]['results']
        res_mean = results['mean_pool'][checkpoint]['results']
        
        ax.plot(res_last, 'b-', alpha=0.7, label='Last Token', linewidth=2)
        ax.plot(res_mean, 'g-', alpha=0.7, label='Mean Pool', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{checkpoint}: Mean Pool vs Last Token")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pooling_comparison.png", dpi=150)
    print(f"\nSaved: {output_dir}/pooling_comparison.png")
    
    # Summary table
    print("\n" + "-"*60)
    print("POOLING COMPARISON SUMMARY")
    print("-"*60)
    print(f"{'Checkpoint':<12} {'Last Token':<15} {'Mean Pool':<15} {'Winner':<12}")
    print("-"*60)
    
    for checkpoint in available_last:
        if checkpoint in results['mean_pool']:
            last = results['last_token'][checkpoint]['best_score']
            mean = results['mean_pool'][checkpoint]['best_score']
            winner = "Mean" if mean > last + 0.01 else ("Last" if last > mean + 0.01 else "Tie")
            print(f"{checkpoint:<12} {last:<15.3f} {mean:<15.3f} {winner:<12}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def train_all_probes(data_file="probe_data.pt", output_dir="probe_results", 
                     run_control=True, run_heatmap=True, run_pooling_comparison=True,
                     run_mlp=False):
    """Main function to train all probe types."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data, num_layers, hidden_dim, has_50pct, has_mean_pooling = load_data(data_file)
    
    # Determine checkpoints
    checkpoints = ['0pct', '100pct']
    if has_50pct:
        checkpoints.insert(1, '50pct')
    
    # If mean pooling is available, we can optionally train on those too
    # For now, we use the default (last_token) activations
    
    all_results = {}
    
    # 1. Success Probes (with diff-of-means)
    all_results['success'] = train_success_probe(data, num_layers, output_dir, checkpoints)
    
    # 2. 2D Heatmap (Layer × Checkpoint)
    if run_heatmap and len(checkpoints) >= 2:
        all_results['heatmap'] = generate_2d_heatmap(data, num_layers, output_dir, checkpoints)
    
    # 3. Control Task (shuffled labels)
    if run_control:
        all_results['control'] = train_control_probe(data, num_layers, output_dir)
    
    # 4. Topic Probes
    all_results['topic'] = train_topic_probe(data, num_layers, output_dir)
    
    # 5. Difficulty Probes
    all_results['difficulty_class'] = train_difficulty_probe(
        data, num_layers, output_dir, mode='classification'
    )
    all_results['difficulty_regress'] = train_difficulty_probe(
        data, num_layers, output_dir, mode='regression'
    )
    
    # 6. Pooling Comparison (if mean pooling available)
    if run_pooling_comparison and has_mean_pooling:
        all_results['pooling_comparison'] = compare_pooling_methods(
            data, num_layers, output_dir, checkpoints
        )
    
    # 7. MLP Probes (non-linear comparison)
    if run_mlp:
        all_results['mlp'] = train_mlp_success_probe(data, num_layers, output_dir, checkpoints)
    
    # 8. PCA Visualizations
    for color_by in ['correctness', 'topic', 'difficulty']:
        generate_pca_visualization(data, num_layers, output_dir, color_by=color_by)
    
    # Save summary
    summary = {
        'num_samples': len(data),
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'has_50pct': has_50pct,
        'has_mean_pooling': has_mean_pooling,
    }
    
    # Add pooling comparison to summary if available
    if 'pooling_comparison' in all_results and all_results['pooling_comparison']:
        pc = all_results['pooling_comparison']
        for checkpoint in pc.get('last_token', {}):
            summary[f'pooling_{checkpoint}_last_token'] = pc['last_token'][checkpoint]['best_score']
            if checkpoint in pc.get('mean_pool', {}):
                summary[f'pooling_{checkpoint}_mean_pool'] = pc['mean_pool'][checkpoint]['best_score']
    
    # Add best results
    for probe_type, results in all_results.items():
        if results and isinstance(results, dict):
            for key, val in results.items():
                if isinstance(val, dict):
                    if 'logreg_best_score' in val:
                        summary[f'{probe_type}_{key}_logreg_best'] = val['logreg_best_score']
                        summary[f'{probe_type}_{key}_diff_best'] = val['diff_best_score']
                    elif 'best_score' in val:
                        summary[f'{probe_type}_{key}_best_layer'] = val['best_layer']
                        summary[f'{probe_type}_{key}_best_score'] = val['best_score']
    
    with open(f"{output_dir}/summary.json", 'w') as f:
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
    parser.add_argument("--no_control", action="store_true",
                        help="Skip control task (shuffled labels)")
    parser.add_argument("--no_heatmap", action="store_true",
                        help="Skip 2D heatmap generation")
    parser.add_argument("--mlp", action="store_true",
                        help="Train MLP probes in addition to linear probes")
    
    args = parser.parse_args()
    
    train_all_probes(
        data_file=args.data_file,
        output_dir=args.output_dir,
        run_control=not args.no_control,
        run_heatmap=not args.no_heatmap,
        run_mlp=args.mlp,
    )
