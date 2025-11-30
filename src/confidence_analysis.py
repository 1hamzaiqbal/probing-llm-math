"""
Confidence Calibration and Progression Analysis.

Analyzes:
1. Probe confidence calibration (reliability diagrams)
2. Confidence trajectories across checkpoints (0% → 50% → 100%)
3. Recovery/degradation patterns
4. Whether confidence change predicts success

Usage:
    python src/confidence_analysis.py --data_file probe_data.pt --output_dir confidence_analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import argparse
import os
import json


def load_probe_data(data_file):
    """Load probe data file."""
    loaded = torch.load(data_file)
    if isinstance(loaded, dict) and 'data' in loaded:
        return loaded['data'], loaded.get('metadata', {})
    return loaded, {}


def train_probe_with_confidence(X_train, y_train, X_test):
    """Train logistic regression and return probabilities."""
    clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (correct)
    return probs, clf


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute calibration metrics."""
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return prob_true, prob_pred, ece


def analyze_confidence(data_file, output_dir="confidence_analysis", target_layer=15):
    """Main analysis function."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_file}...")
    data, metadata = load_probe_data(data_file)
    print(f"Loaded {len(data)} samples")
    
    y = np.array([1 if d['is_correct'] else 0 for d in data])
    num_layers = data[0]['activations_0pct'].shape[0]
    
    # Determine available checkpoints
    checkpoints = ['0pct', '100pct']
    if 'activations_50pct' in data[0]:
        checkpoints = ['0pct', '50pct', '100pct']
    
    print(f"Checkpoints: {checkpoints}")
    print(f"Target layer for analysis: {target_layer}")
    
    # =========================================================================
    # PART 1: Train probes and get confidence for each checkpoint
    # =========================================================================
    print("\n" + "="*60)
    print("PART 1: TRAINING PROBES AND EXTRACTING CONFIDENCE")
    print("="*60)
    
    # Store confidence values for each sample at each checkpoint
    all_confidences = {cp: np.zeros(len(data)) for cp in checkpoints}
    all_predictions = {cp: np.zeros(len(data), dtype=int) for cp in checkpoints}
    calibration_data = {}
    
    for checkpoint in checkpoints:
        key = f'activations_{checkpoint}'
        X = np.array([d[key][target_layer].numpy() for d in data])
        
        # Use leave-one-out style: train on 80%, predict on 20%, repeat
        # For simplicity, we'll do a single split but store all predictions
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(data)), test_size=0.2, random_state=42, stratify=y
        )
        
        # Train probe
        probs_test, clf = train_probe_with_confidence(X_train, y_train, X_test)
        
        # Get predictions for all samples using the trained model
        all_probs = clf.predict_proba(X)[:, 1]
        all_confidences[checkpoint] = all_probs
        all_predictions[checkpoint] = (all_probs > 0.5).astype(int)
        
        # Compute calibration on test set
        prob_true, prob_pred, ece = compute_calibration_metrics(y_test, probs_test)
        calibration_data[checkpoint] = {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece,
            'accuracy': (probs_test > 0.5).astype(int) == y_test
        }
        
        acc = np.mean((all_probs > 0.5).astype(int) == y)
        print(f"  {checkpoint}: Accuracy={acc:.3f}, ECE={ece:.3f}")
    
    # =========================================================================
    # PART 2: CALIBRATION CURVES
    # =========================================================================
    print("\n" + "="*60)
    print("PART 2: CALIBRATION CURVES (RELIABILITY DIAGRAMS)")
    print("="*60)
    
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5*len(checkpoints), 5))
    if len(checkpoints) == 1:
        axes = [axes]
    
    for idx, checkpoint in enumerate(checkpoints):
        ax = axes[idx]
        cd = calibration_data[checkpoint]
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Calibration curve
        ax.plot(cd['prob_pred'], cd['prob_true'], 's-', markersize=8, 
                label=f'Probe (ECE={cd["ece"]:.3f})')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{checkpoint} Checkpoint\nCalibration Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curves.png", dpi=150)
    print(f"Saved: {output_dir}/calibration_curves.png")
    
    # =========================================================================
    # PART 3: CONFIDENCE TRAJECTORIES
    # =========================================================================
    print("\n" + "="*60)
    print("PART 3: CONFIDENCE TRAJECTORIES (0% → 50% → 100%)")
    print("="*60)
    
    if len(checkpoints) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: All trajectories colored by correctness
        ax1 = axes[0]
        
        x_positions = np.arange(len(checkpoints))
        
        for i in range(len(data)):
            conf_trajectory = [all_confidences[cp][i] for cp in checkpoints]
            color = 'green' if y[i] == 1 else 'red'
            alpha = 0.15
            ax1.plot(x_positions, conf_trajectory, color=color, alpha=alpha, linewidth=0.8)
        
        # Mean trajectories
        correct_mask = y == 1
        incorrect_mask = y == 0
        
        mean_correct = [all_confidences[cp][correct_mask].mean() for cp in checkpoints]
        mean_incorrect = [all_confidences[cp][incorrect_mask].mean() for cp in checkpoints]
        
        ax1.plot(x_positions, mean_correct, 'g-', linewidth=3, marker='o', markersize=10,
                label=f'Correct mean (n={correct_mask.sum()})')
        ax1.plot(x_positions, mean_incorrect, 'r-', linewidth=3, marker='o', markersize=10,
                label=f'Incorrect mean (n={incorrect_mask.sum()})')
        
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(checkpoints)
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Probe Confidence (P(correct))')
        ax1.set_title('Confidence Trajectories by Correctness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Confidence change distribution
        ax2 = axes[1]
        
        # Compute confidence change (final - initial)
        conf_change = all_confidences[checkpoints[-1]] - all_confidences[checkpoints[0]]
        
        # Histogram by correctness
        bins = np.linspace(-0.8, 0.8, 30)
        ax2.hist(conf_change[correct_mask], bins=bins, alpha=0.6, label='Correct', color='green')
        ax2.hist(conf_change[incorrect_mask], bins=bins, alpha=0.6, label='Incorrect', color='red')
        
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=conf_change[correct_mask].mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Correct mean: {conf_change[correct_mask].mean():+.2f}')
        ax2.axvline(x=conf_change[incorrect_mask].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Incorrect mean: {conf_change[incorrect_mask].mean():+.2f}')
        
        ax2.set_xlabel(f'Confidence Change ({checkpoints[-1]} - {checkpoints[0]})')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Confidence Change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_trajectories.png", dpi=150)
        print(f"Saved: {output_dir}/confidence_trajectories.png")
        
        # Stats
        print(f"\nConfidence change ({checkpoints[-1]} - {checkpoints[0]}):")
        print(f"  Correct samples: mean={conf_change[correct_mask].mean():+.3f}")
        print(f"  Incorrect samples: mean={conf_change[incorrect_mask].mean():+.3f}")
    
    # =========================================================================
    # PART 4: RECOVERY/DEGRADATION PATTERNS
    # =========================================================================
    print("\n" + "="*60)
    print("PART 4: RECOVERY AND DEGRADATION PATTERNS")
    print("="*60)
    
    if len(checkpoints) >= 2:
        # Categorize samples based on prediction trajectory
        pred_0 = all_predictions[checkpoints[0]]
        pred_final = all_predictions[checkpoints[-1]]
        
        # Categories:
        # - Stable Correct: predicted correct at both, actually correct
        # - Stable Incorrect: predicted incorrect at both, actually incorrect
        # - Recovery: predicted wrong → right, actually correct
        # - Degradation: predicted right → wrong, actually incorrect
        # - False Recovery: predicted wrong → right, actually incorrect
        # - False Degradation: predicted right → wrong, actually correct
        
        categories = []
        for i in range(len(data)):
            p0, pf = pred_0[i], pred_final[i]
            actual = y[i]
            
            if p0 == pf == actual:
                cat = "Stable Correct" if actual == 1 else "Stable Incorrect"
            elif p0 == 0 and pf == 1:  # Prediction went from wrong to right
                cat = "Recovery" if actual == 1 else "False Recovery"
            elif p0 == 1 and pf == 0:  # Prediction went from right to wrong
                cat = "Degradation" if actual == 0 else "False Degradation"
            else:
                cat = "Other"
            
            categories.append(cat)
        
        categories = np.array(categories)
        
        # Count categories
        unique_cats, counts = np.unique(categories, return_counts=True)
        
        print("\nPrediction Trajectory Categories:")
        for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1]):
            pct = 100 * count / len(data)
            print(f"  {cat}: {count} ({pct:.1f}%)")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cat_order = ['Stable Correct', 'Recovery', 'Stable Incorrect', 'Degradation', 
                     'False Recovery', 'False Degradation', 'Other']
        cat_colors = ['green', 'lightgreen', 'red', 'orange', 'pink', 'salmon', 'gray']
        
        cat_counts = {cat: 0 for cat in cat_order}
        for cat in categories:
            if cat in cat_counts:
                cat_counts[cat] += 1
        
        present_cats = [c for c in cat_order if cat_counts[c] > 0]
        present_counts = [cat_counts[c] for c in present_cats]
        present_colors = [cat_colors[cat_order.index(c)] for c in present_cats]
        
        bars = ax.bar(present_cats, present_counts, color=present_colors, edgecolor='black')
        
        # Add percentage labels
        for bar, count in zip(bars, present_counts):
            pct = 100 * count / len(data)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f'{pct:.1f}%', ha='center', fontsize=10)
        
        ax.set_xlabel('Trajectory Category')
        ax.set_ylabel('Count')
        ax.set_title(f'Prediction Trajectory Categories\n({checkpoints[0]} → {checkpoints[-1]})')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trajectory_categories.png", dpi=150)
        print(f"Saved: {output_dir}/trajectory_categories.png")
    
    # =========================================================================
    # PART 5: CONFIDENCE CHANGE VS CORRECTNESS
    # =========================================================================
    print("\n" + "="*60)
    print("PART 5: DOES CONFIDENCE CHANGE PREDICT SUCCESS?")
    print("="*60)
    
    if len(checkpoints) >= 2:
        conf_change = all_confidences[checkpoints[-1]] - all_confidences[checkpoints[0]]
        
        # Correlation
        corr = np.corrcoef(conf_change, y)[0, 1]
        print(f"\nCorrelation (conf_change vs correctness): {corr:.3f}")
        
        # Can we predict correctness from confidence change?
        # Using a simple threshold
        thresholds = np.linspace(-0.5, 0.5, 21)
        best_acc = 0
        best_thresh = 0
        
        for thresh in thresholds:
            pred_from_change = (conf_change > thresh).astype(int)
            acc = np.mean(pred_from_change == y)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        print(f"Best threshold for predicting correctness from Δconfidence: {best_thresh:.2f}")
        print(f"  Accuracy: {best_acc:.3f}")
        
        # Compare to just using final confidence
        pred_from_final = (all_confidences[checkpoints[-1]] > 0.5).astype(int)
        acc_final = np.mean(pred_from_final == y)
        print(f"\nComparison:")
        print(f"  Predict from Δconfidence (>{best_thresh:.2f}): {best_acc:.3f}")
        print(f"  Predict from final confidence (>0.5): {acc_final:.3f}")
        
        # Plot: confidence change vs correctness
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter with jitter for correctness
        jitter = np.random.normal(0, 0.05, len(y))
        ax.scatter(conf_change[y==1], y[y==1] + jitter[y==1], alpha=0.4, c='green', s=30, label='Correct')
        ax.scatter(conf_change[y==0], y[y==0] + jitter[y==0], alpha=0.4, c='red', s=30, label='Incorrect')
        
        # Logistic fit
        from scipy.special import expit
        x_range = np.linspace(conf_change.min(), conf_change.max(), 100)
        
        # Simple logistic regression
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(conf_change.reshape(-1, 1), y)
        y_pred_line = clf.predict_proba(x_range.reshape(-1, 1))[:, 1]
        
        ax.plot(x_range, y_pred_line, 'b-', linewidth=2, label='Logistic fit')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'Confidence Change ({checkpoints[-1]} - {checkpoints[0]})')
        ax.set_ylabel('P(Correct)')
        ax.set_title(f'Confidence Change vs Correctness\nCorrelation: {corr:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_change_prediction.png", dpi=150)
        print(f"Saved: {output_dir}/confidence_change_prediction.png")
    
    # =========================================================================
    # PART 6: LAYER-WISE CALIBRATION COMPARISON
    # =========================================================================
    print("\n" + "="*60)
    print("PART 6: WHICH LAYER IS BEST CALIBRATED?")
    print("="*60)
    
    checkpoint = checkpoints[-1]  # Use final checkpoint
    key = f'activations_{checkpoint}'
    
    layer_eces = []
    layer_accs = []
    
    for layer_idx in range(num_layers):
        X = np.array([d[key][layer_idx].numpy() for d in data])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        probs_test, _ = train_probe_with_confidence(X_train, y_train, X_test)
        _, _, ece = compute_calibration_metrics(y_test, probs_test)
        acc = np.mean((probs_test > 0.5).astype(int) == y_test)
        
        layer_eces.append(ece)
        layer_accs.append(acc)
    
    best_calibrated = np.argmin(layer_eces)
    best_accurate = np.argmax(layer_accs)
    
    print(f"\nBest calibrated layer: {best_calibrated} (ECE={layer_eces[best_calibrated]:.3f})")
    print(f"Most accurate layer: {best_accurate} (Acc={layer_accs[best_accurate]:.3f})")
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(layer_accs, 'b-', linewidth=2, marker='o', markersize=4, label='Accuracy')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.plot(layer_eces, 'r-', linewidth=2, marker='s', markersize=4, label='ECE (lower=better)')
    ax2.set_ylabel('Expected Calibration Error', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Mark best points
    ax1.scatter([best_accurate], [layer_accs[best_accurate]], color='blue', s=150, zorder=5, 
               marker='*', label=f'Best Acc: L{best_accurate}')
    ax2.scatter([best_calibrated], [layer_eces[best_calibrated]], color='red', s=150, zorder=5,
               marker='*', label=f'Best ECE: L{best_calibrated}')
    
    ax1.set_title(f'Layer-wise Accuracy vs Calibration ({checkpoint})')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/layer_calibration.png", dpi=150)
    print(f"Saved: {output_dir}/layer_calibration.png")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = {
        'num_samples': len(data),
        'target_layer': target_layer,
        'checkpoints': checkpoints,
    }
    
    for cp in checkpoints:
        summary[f'{cp}_ece'] = float(calibration_data[cp]['ece'])
    
    if len(checkpoints) >= 2:
        summary['conf_change_correlation'] = float(corr)
        summary['best_change_threshold'] = float(best_thresh)
        summary['best_change_accuracy'] = float(best_acc)
    
    summary['best_calibrated_layer'] = int(best_calibrated)
    summary['best_calibrated_ece'] = float(layer_eces[best_calibrated])
    summary['best_accurate_layer'] = int(best_accurate)
    summary['best_accurate_acc'] = float(layer_accs[best_accurate])
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_dir}/summary.json")
    
    print("\nKey Findings:")
    for cp in checkpoints:
        print(f"  {cp} ECE: {calibration_data[cp]['ece']:.3f}")
    
    if len(checkpoints) >= 2:
        print(f"  Confidence change correlation: {corr:.3f}")
        print(f"  Best calibrated layer: {best_calibrated}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence calibration and progression analysis")
    parser.add_argument("--data_file", type=str, default="probe_data.pt",
                        help="Path to probe data file")
    parser.add_argument("--output_dir", type=str, default="confidence_analysis",
                        help="Output directory for plots")
    parser.add_argument("--layer", type=int, default=15,
                        help="Target layer for detailed analysis")
    
    args = parser.parse_args()
    
    analyze_confidence(args.data_file, args.output_dir, args.layer)

