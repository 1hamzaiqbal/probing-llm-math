import re
from sympy import simplify, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def extract_answer(text):
    """
    Extracts the final answer from the model output.
    """
    text = text.strip()
    
    # 1. Look for \boxed{...}
    simple_boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if simple_boxed:
        return simple_boxed[-1].strip()
        
    if "\\boxed{" in text:
        start_idx = text.rfind("\\boxed{") + 7
        balance = 1
        end_idx = start_idx
        while end_idx < len(text) and balance > 0:
            if text[end_idx] == '{':
                balance += 1
            elif text[end_idx] == '}':
                balance -= 1
            end_idx += 1
        
        if balance == 0:
            return text[start_idx:end_idx-1].strip()

    # 3. GSM8K style "####"
    if "####" in text:
        return text.split("####")[-1].strip()
        
    # 4. Fallback
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
    return lines[-1]

def is_equivalent(model_ans, ground_truth):
    """
    Checks if model_ans is mathematically equivalent to ground_truth.
    """
    # 1. Exact string match (normalized)
    if model_ans.strip() == ground_truth.strip():
        return True
        
    # 2. Numeric comparison
    try:
        num_model = float(model_ans)
        num_gt = float(ground_truth)
        if abs(num_model - num_gt) < 1e-6:
            return True
    except ValueError:
        pass
        
    # 3. SymPy equivalence
    try:
        # Setup transformations for implicit multiplication (e.g. "2x" -> "2*x")
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        # Parse expressions
        expr_model = parse_expr(model_ans, transformations=transformations)
        expr_gt = parse_expr(ground_truth, transformations=transformations)
        
        # Check if difference simplifies to 0
        diff = simplify(expr_model - expr_gt)
        if diff == 0:
            return True
    except Exception:
        pass
        
    return False
