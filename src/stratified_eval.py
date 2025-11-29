"""
Stratified Evaluation Script

Runs the model on a stratified sample of problems and generates:
1. Accuracy heatmap (topic × difficulty)
2. Performance summary table
3. Identifies "cliffs" (steep accuracy drops)
"""

import torch
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

import model_utils
import evaluator


# Topic configurations
TOPICS = ['algebra', 'number_theory', 'counting_and_probability', 'precalculus']
LEVELS = [1, 2, 3, 4, 5]

TOPIC_DISPLAY = {
    'algebra': 'Algebra',
    'number_theory': 'Number Theory',
    'counting_and_probability': 'Counting/Prob',
    'precalculus': 'Precalculus',
    'intermediate_algebra': 'Int. Algebra',
    'prealgebra': 'Pre-Algebra',
}


def load_stratified_dataset(samples_per_cell=5, topics=None, levels=None, exclude_geometry=True):
    """
    Load a stratified sample from MATH dataset.
    
    Args:
        samples_per_cell: Number of problems per (topic, level) combination
        topics: List of topics to include (None = all)
        levels: List of levels to include (None = all 1-5)
        
    Returns:
        List of problem dicts with topic, level, problem, solution
    """
    from datasets import load_dataset
    
    if topics is None:
        topics = TOPICS
    if levels is None:
        levels = LEVELS
    
    if exclude_geometry and 'geometry' in topics:
        topics = [t for t in topics if t != 'geometry']
    
    all_problems = []
    
    print("Loading stratified dataset...")
    
    for topic in topics:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", topic, split="test")
            
            # Group by level
            by_level = defaultdict(list)
            for item in ds:
                level_str = item.get('level', 'Level 1')
                level_num = int(level_str.replace('Level ', ''))
                if level_num in levels:
                    by_level[level_num].append(item)
            
            # Sample from each level
            for level in levels:
                available = by_level[level]
                if len(available) == 0:
                    print(f"  Warning: No problems for {topic} Level {level}")
                    continue
                
                # Random sample
                np.random.shuffle(available)
                sampled = available[:samples_per_cell]
                
                for item in sampled:
                    all_problems.append({
                        'topic': topic,
                        'level': level,
                        'problem': item['problem'],
                        'solution': item['solution'],
                    })
                    
        except Exception as e:
            print(f"Error loading {topic}: {e}")
    
    print(f"Loaded {len(all_problems)} problems")
    
    # Print distribution
    dist = defaultdict(lambda: defaultdict(int))
    for p in all_problems:
        dist[p['topic']][p['level']] += 1
    
    print("\nProblem distribution:")
    for topic in topics:
        counts = [dist[topic].get(l, 0) for l in levels]
        print(f"  {topic}: {counts}")
    
    return all_problems


def run_evaluation(problems, output_dir="eval_results", save_interval=10):
    """
    Run model on all problems and collect results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = model_utils.load_model(load_in_4bit=True)
    
    results = []
    
    # Prepare CSV log
    csv_path = os.path.join(output_dir, "eval_log.csv")
    csv_fields = ['idx', 'topic', 'level', 'question', 'ground_truth', 
                  'prediction', 'clean_pred', 'is_correct']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
    
    print(f"\nEvaluating {len(problems)} problems...")
    
    for i, prob in enumerate(tqdm(problems)):
        question = prob['problem']
        topic = prob['topic']
        level = prob['level']
        
        # Extract ground truth
        ground_truth = evaluator.extract_answer(prob['solution'])
        
        # Generate
        pred_text, _, is_truncated = model_utils.generate_answer(model, tokenizer, question)
        
        # Evaluate
        clean_pred = evaluator.extract_answer(pred_text)
        is_correct = evaluator.is_equivalent(clean_pred, ground_truth)
        
        result = {
            'idx': i,
            'topic': topic,
            'level': level,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': pred_text,
            'clean_pred': clean_pred,
            'is_correct': is_correct,
        }
        results.append(result)
        
        # Write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            row = {k: v if k != 'question' and k != 'prediction' else v[:200] 
                   for k, v in result.items()}
            writer.writerow(row)
        
        # Periodic save
        if (i + 1) % save_interval == 0:
            torch.save(results, os.path.join(output_dir, "results.pt"))
    
    torch.save(results, os.path.join(output_dir, "results.pt"))
    print(f"\nSaved {len(results)} results to {output_dir}/results.pt")
    
    return results


def compute_accuracy_matrix(results, topics=None, levels=None):
    """
    Compute accuracy matrix from results.
    
    Returns:
        accuracy_matrix: numpy array of shape (num_topics, num_levels)
        count_matrix: numpy array with sample counts per cell
    """
    if topics is None:
        topics = sorted(set(r['topic'] for r in results))
    if levels is None:
        levels = sorted(set(r['level'] for r in results))
    
    # Initialize matrices
    correct_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        topic = r['topic']
        level = r['level']
        total_counts[topic][level] += 1
        if r['is_correct']:
            correct_counts[topic][level] += 1
    
    # Build matrices
    accuracy_matrix = np.zeros((len(topics), len(levels)))
    count_matrix = np.zeros((len(topics), len(levels)), dtype=int)
    
    for i, topic in enumerate(topics):
        for j, level in enumerate(levels):
            total = total_counts[topic][level]
            correct = correct_counts[topic][level]
            count_matrix[i, j] = total
            if total > 0:
                accuracy_matrix[i, j] = correct / total
            else:
                accuracy_matrix[i, j] = np.nan  # No data
    
    return accuracy_matrix, count_matrix, topics, levels


def plot_heatmap(accuracy_matrix, count_matrix, topics, levels, output_path):
    """
    Generate and save the accuracy heatmap.
    """
    # Create display names
    topic_labels = [TOPIC_DISPLAY.get(t, t) for t in topics]
    level_labels = [f"Level {l}" for l in levels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    mask = np.isnan(accuracy_matrix)
    
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        mask=mask,
        xticklabels=level_labels,
        yticklabels=topic_labels,
        cbar_kws={'label': 'Accuracy'},
        ax=ax,
        linewidths=0.5,
    )
    
    # Add count annotations in smaller text
    for i in range(len(topics)):
        for j in range(len(levels)):
            if not mask[i, j]:
                count = count_matrix[i, j]
                ax.text(j + 0.5, i + 0.75, f'n={count}', 
                       ha='center', va='center', fontsize=8, color='gray')
    
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)
    ax.set_title('Model Accuracy: Topic × Difficulty', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def identify_cliffs(accuracy_matrix, topics, levels, threshold=0.2):
    """
    Identify "cliffs" where accuracy drops sharply.
    
    A cliff is defined as a drop of more than `threshold` between adjacent levels.
    """
    cliffs = []
    
    for i, topic in enumerate(topics):
        for j in range(len(levels) - 1):
            acc_current = accuracy_matrix[i, j]
            acc_next = accuracy_matrix[i, j + 1]
            
            if np.isnan(acc_current) or np.isnan(acc_next):
                continue
            
            drop = acc_current - acc_next
            if drop >= threshold:
                cliffs.append({
                    'topic': topic,
                    'from_level': levels[j],
                    'to_level': levels[j + 1],
                    'from_acc': acc_current,
                    'to_acc': acc_next,
                    'drop': drop,
                })
    
    # Sort by drop magnitude
    cliffs.sort(key=lambda x: x['drop'], reverse=True)
    
    return cliffs


def generate_summary(results, accuracy_matrix, count_matrix, topics, levels, cliffs, output_dir):
    """
    Generate a text summary of the evaluation.
    """
    summary_path = os.path.join(output_dir, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STRATIFIED EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall stats
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        f.write(f"Total problems: {total}\n")
        f.write(f"Correct: {correct} ({100*correct/total:.1f}%)\n\n")
        
        # Per-topic accuracy
        f.write("Accuracy by Topic:\n")
        for topic in topics:
            topic_results = [r for r in results if r['topic'] == topic]
            if topic_results:
                acc = sum(1 for r in topic_results if r['is_correct']) / len(topic_results)
                f.write(f"  {TOPIC_DISPLAY.get(topic, topic)}: {100*acc:.1f}%\n")
        
        f.write("\n")
        
        # Per-level accuracy
        f.write("Accuracy by Level:\n")
        for level in levels:
            level_results = [r for r in results if r['level'] == level]
            if level_results:
                acc = sum(1 for r in level_results if r['is_correct']) / len(level_results)
                f.write(f"  Level {level}: {100*acc:.1f}%\n")
        
        f.write("\n")
        
        # Cliffs
        if cliffs:
            f.write("Performance Cliffs (drops > 20%):\n")
            for cliff in cliffs[:5]:  # Top 5
                f.write(f"  {TOPIC_DISPLAY.get(cliff['topic'], cliff['topic'])}: "
                       f"L{cliff['from_level']}→L{cliff['to_level']} "
                       f"({100*cliff['from_acc']:.0f}%→{100*cliff['to_acc']:.0f}%, "
                       f"drop={100*cliff['drop']:.0f}%)\n")
        else:
            f.write("No significant performance cliffs detected.\n")
        
        f.write("\n")
        
        # Accuracy matrix
        f.write("Full Accuracy Matrix:\n")
        f.write("      " + " ".join(f"L{l:>5}" for l in levels) + "\n")
        for i, topic in enumerate(topics):
            row = " ".join(f"{100*accuracy_matrix[i,j]:5.1f}%" if not np.isnan(accuracy_matrix[i,j]) else "  N/A" 
                          for j in range(len(levels)))
            f.write(f"{TOPIC_DISPLAY.get(topic, topic)[:6]:>6} {row}\n")
    
    print(f"Saved summary to {summary_path}")


def run_stratified_eval(
    samples_per_cell=5,
    topics=None,
    levels=None,
    output_dir="eval_results",
    results_file=None,
):
    """
    Main function to run stratified evaluation.
    
    Args:
        samples_per_cell: Number of problems per (topic, level) combination
        topics: List of topics (None = default set)
        levels: List of levels (None = 1-5)
        output_dir: Output directory
        results_file: If provided, load existing results instead of running eval
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if results_file and os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        results = torch.load(results_file)
    else:
        # Load dataset
        problems = load_stratified_dataset(
            samples_per_cell=samples_per_cell,
            topics=topics,
            levels=levels,
        )
        
        # Run evaluation
        results = run_evaluation(problems, output_dir=output_dir)
    
    # Compute accuracy matrix
    accuracy_matrix, count_matrix, topics_used, levels_used = compute_accuracy_matrix(
        results, topics=topics, levels=levels
    )
    
    # Generate heatmap
    heatmap_path = os.path.join(output_dir, "accuracy_heatmap.png")
    plot_heatmap(accuracy_matrix, count_matrix, topics_used, levels_used, heatmap_path)
    
    # Identify cliffs
    cliffs = identify_cliffs(accuracy_matrix, topics_used, levels_used)
    
    # Generate summary
    generate_summary(results, accuracy_matrix, count_matrix, topics_used, levels_used, cliffs, output_dir)
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    print(f"Overall accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    
    if cliffs:
        print(f"\nTop cliff: {cliffs[0]['topic']} L{cliffs[0]['from_level']}→L{cliffs[0]['to_level']} "
              f"(drop: {100*cliffs[0]['drop']:.0f}%)")
    
    return results, accuracy_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stratified evaluation for heatmap generation.")
    parser.add_argument("--samples_per_cell", type=int, default=5,
                        help="Number of samples per (topic, level) cell")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Load existing results file instead of running eval")
    
    args = parser.parse_args()
    
    run_stratified_eval(
        samples_per_cell=args.samples_per_cell,
        output_dir=args.output_dir,
        results_file=args.results_file,
    )

