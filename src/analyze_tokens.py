"""
Analyze relationship between response length (tokens) and correctness.

Usage:
    python src/analyze_tokens.py --audit_file probe_audit.csv --output_dir token_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def analyze_tokens(audit_file, output_dir="token_analysis"):
    """Analyze token counts vs correctness."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(audit_file)
    print(f"Loaded {len(df)} samples from {audit_file}")
    
    # Basic stats
    correct = df[df['is_correct'] == True]
    incorrect = df[df['is_correct'] == False]
    
    print("\n" + "="*60)
    print("TOKEN COUNT STATISTICS")
    print("="*60)
    
    print(f"\nOverall:")
    print(f"  Mean tokens: {df['num_tokens'].mean():.1f}")
    print(f"  Median tokens: {df['num_tokens'].median():.1f}")
    print(f"  Std tokens: {df['num_tokens'].std():.1f}")
    print(f"  Min: {df['num_tokens'].min()}, Max: {df['num_tokens'].max()}")
    
    print(f"\nCorrect answers (n={len(correct)}):")
    print(f"  Mean tokens: {correct['num_tokens'].mean():.1f}")
    print(f"  Median tokens: {correct['num_tokens'].median():.1f}")
    print(f"  Std tokens: {correct['num_tokens'].std():.1f}")
    
    print(f"\nIncorrect answers (n={len(incorrect)}):")
    print(f"  Mean tokens: {incorrect['num_tokens'].mean():.1f}")
    print(f"  Median tokens: {incorrect['num_tokens'].median():.1f}")
    print(f"  Std tokens: {incorrect['num_tokens'].std():.1f}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(correct['num_tokens'], incorrect['num_tokens'])
    print(f"\nT-test (correct vs incorrect):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  → Significant difference!")
    else:
        print("  → No significant difference")
    
    # =========================================================================
    # PLOT 1: Distribution of token counts by correctness
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, df['num_tokens'].quantile(0.95), 30)
    ax1.hist(correct['num_tokens'], bins=bins, alpha=0.6, label=f'Correct (n={len(correct)})', color='green')
    ax1.hist(incorrect['num_tokens'], bins=bins, alpha=0.6, label=f'Incorrect (n={len(incorrect)})', color='red')
    ax1.axvline(correct['num_tokens'].mean(), color='green', linestyle='--', linewidth=2, label=f'Correct mean: {correct["num_tokens"].mean():.0f}')
    ax1.axvline(incorrect['num_tokens'].mean(), color='red', linestyle='--', linewidth=2, label=f'Incorrect mean: {incorrect["num_tokens"].mean():.0f}')
    ax1.set_xlabel('Number of Tokens')
    ax1.set_ylabel('Count')
    ax1.set_title('Token Count Distribution by Correctness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    df['Correctness'] = df['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    sns.boxplot(data=df, x='Correctness', y='num_tokens', ax=ax2, palette=['green', 'red'])
    ax2.set_ylabel('Number of Tokens')
    ax2.set_title('Token Count by Correctness')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_distribution.png", dpi=150)
    print(f"\nSaved: {output_dir}/token_distribution.png")
    
    # =========================================================================
    # PLOT 2: Accuracy by token count bins
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bins
    df['token_bin'] = pd.cut(df['num_tokens'], bins=10, labels=False)
    bin_stats = df.groupby('token_bin').agg({
        'is_correct': ['mean', 'count'],
        'num_tokens': ['min', 'max', 'mean']
    }).reset_index()
    bin_stats.columns = ['bin', 'accuracy', 'count', 'min_tokens', 'max_tokens', 'mean_tokens']
    
    # Bar plot
    bars = ax.bar(range(len(bin_stats)), bin_stats['accuracy'], 
                  color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add count labels
    for i, (acc, count) in enumerate(zip(bin_stats['accuracy'], bin_stats['count'])):
        ax.text(i, acc + 0.02, f'n={count}', ha='center', fontsize=9)
    
    # X-axis labels
    labels = [f"{int(row['min_tokens'])}-{int(row['max_tokens'])}" for _, row in bin_stats.iterrows()]
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axhline(y=df['is_correct'].mean(), color='orange', linestyle='--', alpha=0.7, 
               label=f'Overall: {df["is_correct"].mean():.1%}')
    
    ax.set_xlabel('Token Count Range')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Response Length')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_tokens.png", dpi=150)
    print(f"Saved: {output_dir}/accuracy_by_tokens.png")
    
    # =========================================================================
    # PLOT 3: Token count by topic and difficulty
    # =========================================================================
    if 'topic' in df.columns and 'level' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # By topic
        ax1 = axes[0]
        topic_stats = df.groupby(['topic', 'is_correct'])['num_tokens'].mean().unstack()
        topic_stats.plot(kind='bar', ax=ax1, color=['red', 'green'], alpha=0.7)
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Mean Tokens')
        ax1.set_title('Mean Token Count by Topic and Correctness')
        ax1.legend(['Incorrect', 'Correct'])
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # By level
        ax2 = axes[1]
        level_stats = df.groupby(['level', 'is_correct'])['num_tokens'].mean().unstack()
        level_stats.plot(kind='bar', ax=ax2, color=['red', 'green'], alpha=0.7)
        ax2.set_xlabel('Difficulty Level')
        ax2.set_ylabel('Mean Tokens')
        ax2.set_title('Mean Token Count by Difficulty and Correctness')
        ax2.legend(['Incorrect', 'Correct'])
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tokens_by_topic_level.png", dpi=150)
        print(f"Saved: {output_dir}/tokens_by_topic_level.png")
    
    # =========================================================================
    # PLOT 4: Scatter with trend
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Rolling accuracy (sorted by token count)
    df_sorted = df.sort_values('num_tokens')
    window = min(50, len(df) // 10)
    df_sorted['rolling_acc'] = df_sorted['is_correct'].rolling(window, center=True).mean()
    
    # Scatter
    ax.scatter(correct['num_tokens'], [1]*len(correct), alpha=0.3, c='green', s=20, label='Correct')
    ax.scatter(incorrect['num_tokens'], [0]*len(incorrect), alpha=0.3, c='red', s=20, label='Incorrect')
    
    # Trend line
    ax.plot(df_sorted['num_tokens'], df_sorted['rolling_acc'], 'b-', linewidth=2, 
            label=f'Rolling accuracy (window={window})')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Tokens')
    ax.set_ylabel('Accuracy / Correctness')
    ax.set_title('Correctness vs Token Count with Rolling Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_trend.png", dpi=150)
    print(f"Saved: {output_dir}/token_trend.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find optimal range
    best_bin = bin_stats.loc[bin_stats['accuracy'].idxmax()]
    print(f"\nBest accuracy range: {int(best_bin['min_tokens'])}-{int(best_bin['max_tokens'])} tokens")
    print(f"  Accuracy: {best_bin['accuracy']:.1%} (n={int(best_bin['count'])})")
    
    # Very short / very long
    short_threshold = df['num_tokens'].quantile(0.1)
    long_threshold = df['num_tokens'].quantile(0.9)
    
    short = df[df['num_tokens'] <= short_threshold]
    long = df[df['num_tokens'] >= long_threshold]
    
    print(f"\nShort responses (≤{short_threshold:.0f} tokens):")
    print(f"  Accuracy: {short['is_correct'].mean():.1%} (n={len(short)})")
    
    print(f"\nLong responses (≥{long_threshold:.0f} tokens):")
    print(f"  Accuracy: {long['is_correct'].mean():.1%} (n={len(long)})")
    
    # Correlation
    corr = df['num_tokens'].corr(df['is_correct'].astype(int))
    print(f"\nCorrelation (tokens vs correct): {corr:.3f}")
    
    return {
        'mean_correct': correct['num_tokens'].mean(),
        'mean_incorrect': incorrect['num_tokens'].mean(),
        'p_value': p_value,
        'correlation': corr,
        'best_range': (best_bin['min_tokens'], best_bin['max_tokens']),
        'best_accuracy': best_bin['accuracy']
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token counts vs correctness")
    parser.add_argument("--audit_file", type=str, default="probe_audit.csv",
                        help="Path to audit CSV file")
    parser.add_argument("--output_dir", type=str, default="token_analysis",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    results = analyze_tokens(args.audit_file, args.output_dir)

