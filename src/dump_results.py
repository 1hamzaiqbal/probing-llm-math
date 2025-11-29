#!/usr/bin/env python3
"""
Dump all results to a single text file for easy sharing/debugging.

Creates a comprehensive summary that can be copy-pasted to an AI for iterative development.

Usage:
    python src/dump_results.py --probe_dir probe_results --eval_dir eval_results --output results_dump.txt
"""

import os
import json
import argparse
from datetime import datetime


def read_file_safe(path):
    """Read file if it exists, else return placeholder."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return f"[File not found: {path}]"


def read_json_safe(path):
    """Read JSON file if it exists."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def dump_results(probe_dir="probe_results", eval_dir="eval_results", 
                 audit_file="probe_audit.csv", output="results_dump.txt"):
    """Generate comprehensive results dump."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("PROBING-LLM-MATH RESULTS DUMP")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    # 1. Probe Training Summary
    lines.append("=" * 70)
    lines.append("SECTION 1: PROBE TRAINING RESULTS")
    lines.append("=" * 70)
    
    summary = read_json_safe(f"{probe_dir}/summary.json")
    if summary:
        lines.append("\n### Summary JSON ###")
        lines.append(json.dumps(summary, indent=2))
    else:
        lines.append(f"\n[No summary.json found in {probe_dir}]")
    
    # 2. Stratified Eval Summary
    lines.append("\n" + "=" * 70)
    lines.append("SECTION 2: STRATIFIED EVALUATION")
    lines.append("=" * 70)
    
    eval_summary = read_file_safe(f"{eval_dir}/summary.txt")
    lines.append("\n### Eval Summary ###")
    lines.append(eval_summary)
    
    # 3. Recent Audit Log (first 20 rows for context)
    lines.append("\n" + "=" * 70)
    lines.append("SECTION 3: PROBE DATA AUDIT (First 20 samples)")
    lines.append("=" * 70)
    
    if os.path.exists(audit_file):
        with open(audit_file, 'r') as f:
            audit_lines = f.readlines()[:21]  # Header + 20 rows
            lines.append("\n" + "".join(audit_lines))
    else:
        lines.append(f"\n[No audit file found: {audit_file}]")
    
    # 4. Key Metrics Summary Table
    lines.append("\n" + "=" * 70)
    lines.append("SECTION 4: KEY METRICS TABLE")
    lines.append("=" * 70)
    
    if summary:
        lines.append("\n| Metric | Value |")
        lines.append("|--------|-------|")
        
        for key, val in summary.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.3f} |")
            else:
                lines.append(f"| {key} | {val} |")
    
    # 5. Files Generated
    lines.append("\n" + "=" * 70)
    lines.append("SECTION 5: FILES GENERATED")
    lines.append("=" * 70)
    
    for dir_name, dir_path in [("Probe Results", probe_dir), ("Eval Results", eval_dir)]:
        lines.append(f"\n### {dir_name} ({dir_path}/) ###")
        if os.path.exists(dir_path):
            for f in sorted(os.listdir(dir_path)):
                size = os.path.getsize(os.path.join(dir_path, f))
                lines.append(f"  - {f} ({size/1024:.1f} KB)")
        else:
            lines.append(f"  [Directory not found]")
    
    # 6. Recommendations / Next Steps
    lines.append("\n" + "=" * 70)
    lines.append("SECTION 6: AUTOMATED RECOMMENDATIONS")
    lines.append("=" * 70)
    
    recommendations = []
    
    if summary:
        n = summary.get('num_samples', 0)
        if n < 50:
            recommendations.append(f"⚠️  Only {n} samples collected. Recommend 100+ for reliable probing.")
        
        # Check for low success probe accuracy
        for key in summary:
            if 'success' in key and 'best' in key and isinstance(summary[key], float):
                if summary[key] < 0.6:
                    recommendations.append(f"⚠️  {key} = {summary[key]:.2f} is low. May need more data.")
        
        # Check for control task (if exists)
        control_mean = summary.get('control_mean')
        if control_mean and control_mean > 0.65:
            recommendations.append(f"⚠️  Control task accuracy ({control_mean:.2f}) is high. May be overfitting.")
    
    if not recommendations:
        recommendations.append("✓ No major issues detected.")
    
    for rec in recommendations:
        lines.append(f"  {rec}")
    
    # Write output
    output_text = "\n".join(lines)
    
    with open(output, 'w') as f:
        f.write(output_text)
    
    print(f"Results dump saved to: {output}")
    print(f"Total length: {len(output_text)} chars, {len(lines)} lines")
    print("\nYou can copy this file to share with AI for iterative development.")
    
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump all results to text file.")
    parser.add_argument("--probe_dir", type=str, default="probe_results",
                        help="Directory with probe training results")
    parser.add_argument("--eval_dir", type=str, default="eval_results",
                        help="Directory with stratified eval results")
    parser.add_argument("--audit_file", type=str, default="probe_audit.csv",
                        help="Path to probe audit CSV")
    parser.add_argument("--output", type=str, default="results_dump.txt",
                        help="Output text file")
    
    args = parser.parse_args()
    
    dump_results(
        probe_dir=args.probe_dir,
        eval_dir=args.eval_dir,
        audit_file=args.audit_file,
        output=args.output,
    )

