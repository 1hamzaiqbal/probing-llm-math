#!/usr/bin/env python3
"""
Test script for the improved evaluator.
Run this to verify the fixes for the false negatives we identified.
"""

import sys
sys.path.insert(0, 'src')

from evaluator import is_equivalent, debug_comparison

# Test cases from the failed evaluation
test_cases = [
    # (model_answer, ground_truth, expected_result, description)
    
    # Row 4: Fraction notation in matrix (should be True)
    (
        r"\begin{pmatrix} \frac{2}{5} \\ -\frac{1}{5} \\ 0 \end{pmatrix}",
        r"\begin{pmatrix} 2/5 \\ -1/5 \\ 0 \end{pmatrix}",
        True,
        "Matrix with fraction notation: \\frac{2}{5} vs 2/5"
    ),
    
    # Row 16: Rationalized denominator (should be True)
    (
        r"\begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ -\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \end{pmatrix}",
        r"\begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ -1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}",
        True,
        "Matrix with rationalized denominators: sqrt(2)/2 vs 1/sqrt(2)"
    ),
    
    # Row 17: Complex number ordering (should be True)
    (
        r"\frac{1}{5}i",
        r"\frac{i}{5}",
        True,
        "Complex number: (1/5)i vs i/5"
    ),
    
    # True positives (should remain True)
    (r"-99", r"-99", True, "Simple integer match"),
    (r"11", r"11", True, "Simple integer match"),
    (r"9z(z^2 - 3z + 3)", r"9z(z^2 - 3z + 3)", True, "Polynomial match"),
    (r"\sqrt{10}", r"\sqrt{10}", True, "Square root match"),
    (r"\sqrt{13}", r"\sqrt{13}", True, "Square root match"),
    
    # True negatives (should remain False)
    (r"f(x) = x^3 + 1", r"676", False, "Genuinely wrong answer"),
    (r"22", r"48", False, "Wrong numeric answer"),
    (r"7", r"6", False, "Wrong numeric answer"),
    (r"17", r"11", False, "Wrong numeric answer"),
    
    # Additional edge cases
    (r"\frac{1}{2}", r"0.5", True, "Fraction equals decimal"),
    (r"\frac{\sqrt{2}}{2}", r"1/\sqrt{2}", True, "Rationalized denominator"),
    (r"2i", r"2i", True, "Complex number match"),
    (r"\frac{3}{4}", r"3/4", True, "Fraction notation"),
    
    # More complex cases
    (r"3 + 2i", r"3+2i", True, "Complex number with spaces"),
    (r"\frac{-1}{2}", r"-\frac{1}{2}", True, "Negative fraction"),
    (r"\sqrt{2} + 1", r"1 + \sqrt{2}", True, "Commutative addition"),
    (r"\frac{2\sqrt{3}}{3}", r"\frac{2}{\sqrt{3}}", True, "Rationalized with coefficient"),
    (r"x^2", r"x^2", True, "Variable expression"),
    (r"\pi", r"3.14159265358979", True, "Pi equals its decimal (math equivalence)"),
    
    # Interval/set notation (these should match strings)
    (r"[1, \infty)", r"[1, \infty)", True, "Interval notation match"),
    (r"(1, 1)", r"(1, 1)", True, "Coordinate tuple match"),
    
    # Edge cases that should fail
    (r"\frac{1}{3}", r"0.33", False, "Fraction vs truncated decimal"),
    (r"2 + 3i", r"3 + 2i", False, "Different complex numbers"),
    
    # === NEW TEST CASES ===
    
    # Coordinate tuples with mixed fraction/decimal (the Row 0 issue)
    (r"(1, 4.5)", r"(1,\frac{9}{2})", True, "Tuple: decimal vs fraction"),
    (r"(1, 4.5)", r"(1, \frac{9}{2})", True, "Tuple: decimal vs fraction with space"),
    (r"(-3, 2)", r"(-3,2)", True, "Tuple: spacing difference"),
    (r"(0, 0)", r"(0, 0)", True, "Tuple: origin"),
    (r"(\frac{1}{2}, \frac{3}{4})", r"(0.5, 0.75)", True, "Tuple: both elements fractions"),
    
    # Numbers with thousands separators
    (r"90900909", r"90,900,909", True, "Thousands separators"),
    (r"1000000", r"1,000,000", True, "Large number with commas"),
    (r"123456789", r"123,456,789", True, "Even larger number"),
    
    # Set notation
    (r"\{1, 2, 3\}", r"\{1, 2, 3\}", True, "Set: exact match"),
    (r"{1, 2, 3}", r"\{1, 2, 3\}", True, "Set: LaTeX vs plain braces"),
    (r"\{3, 1, 2\}", r"\{1, 2, 3\}", True, "Set: different order (sets are unordered)"),
    
    # More interval cases
    (r"[1, \infty)", r"[1, infty)", True, "Interval: different infinity notation"),
    (r"(-\infty, 5]", r"(-infty, 5]", True, "Interval: negative infinity"),
    
    # Scientific notation (basic)
    (r"1000", r"10^3", True, "Scientific: power of 10"),
    
    # Negative numbers in various formats
    (r"-\frac{1}{2}", r"-0.5", True, "Negative fraction vs decimal"),
    (r"(-1, -2)", r"(-1,-2)", True, "Tuple: negative values"),
    
    # Verify false negatives don't become false positives
    (r"(1, 2)", r"(2, 1)", False, "Tuple: order matters"),
    (r"90900909", r"100899919", False, "Different large numbers"),
    (r"\{1, 2\}", r"\{1, 2, 3\}", False, "Set: different sizes"),
    
    # Base notation (subscript formatting)
    (r"4210_{5}", r"4210_5", True, "Base notation: braces vs no braces"),
    (r"4210_{7}", r"4210_7", True, "Base notation: braces vs no braces"),
    (r"101_{2}", r"101_2", True, "Binary base notation"),
]


def run_tests():
    print("=" * 70)
    print("EVALUATOR TEST SUITE")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    for model_ans, ground_truth, expected, description in test_cases:
        result = is_equivalent(model_ans, ground_truth)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
            print(f"{status}: {description}")
        else:
            failed += 1
            print(f"{status}: {description}")
            print(f"       Model:    {model_ans}")
            print(f"       Truth:    {ground_truth}")
            print(f"       Expected: {expected}, Got: {result}")
            print()
            # Debug the failing case
            print("       --- Debug output ---")
            debug_comparison(model_ans, ground_truth)
            print()
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    return failed == 0


def test_specific_case(model_ans, ground_truth):
    """Test a specific case with debug output."""
    print("Testing specific case:")
    print(f"  Model answer: {model_ans}")
    print(f"  Ground truth: {ground_truth}")
    print()
    return debug_comparison(model_ans, ground_truth)


if __name__ == "__main__":
    # Run all tests
    success = run_tests()
    
    # If there are command line arguments, test those
    if len(sys.argv) >= 3:
        print("\n" + "=" * 70)
        print("CUSTOM TEST")
        print("=" * 70)
        test_specific_case(sys.argv[1], sys.argv[2])
    
    sys.exit(0 if success else 1)

