from src.evaluator import is_equivalent

def test_case(gold, pred, name):
    result = is_equivalent(pred, gold)
    print(f"Test '{name}': Gold='{gold}', Pred='{pred}' -> {result}")

if __name__ == "__main__":
    test_case("-2", "-2", "Negative Two")
    test_case("0", "0", "Zero")
    test_case("1+\\sqrt{2}", "1 + \\sqrt{2}", "LaTeX Space")
    test_case("\\frac{1}{2}", "\\frac{1}{2}", "Fraction")
