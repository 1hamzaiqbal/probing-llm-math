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

def normalize_latex(text):
    """
    Simple text normalization to convert LaTeX math to SymPy-friendly format.
    """
    # Remove \left and \right
    text = text.replace(r'\left', '').replace(r'\right', '')
    
    # Replace \frac{a}{b} with (a)/(b)
    # Note: This handles simple non-nested fractions.
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
    
    # Replace \sqrt{x} with sqrt(x)
    text = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', text)
    
    # Replace \cdot and \times with *
    text = text.replace(r'\cdot', '*').replace(r'\times', '*')
    
    return text

def is_equivalent(model_ans, ground_truth):
    """
    Checks if model_ans is mathematically equivalent to ground_truth.
    """
    # 1. Exact string match (normalized)
    if model_ans.strip() == ground_truth.strip():
        return True
    
    # Normalize LaTeX for further checks
    model_ans_norm = normalize_latex(model_ans)
    ground_truth_norm = normalize_latex(ground_truth)
        
    # 2. Numeric comparison
    try:
        # Try to eval simple arithmetic strings like "5/6"
        # eval() is unsafe generally, but here we are in a controlled env. 
        # Better to use sympify which handles "5/6" -> Rational(5, 6)
        num_model = float(sympify(model_ans_norm))
        num_gt = float(sympify(ground_truth_norm))
        if abs(num_model - num_gt) < 1e-6:
            return True
    except Exception:
        pass
        
    # 3. SymPy equivalence
    try:
        # Setup transformations for implicit multiplication (e.g. "2x" -> "2*x")
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        # Parse expressions
        expr_model = parse_expr(model_ans_norm, transformations=transformations)
        expr_gt = parse_expr(ground_truth_norm, transformations=transformations)
        
        # Check if difference simplifies to 0
        diff = simplify(expr_model - expr_gt)
        if diff == 0:
            return True
    except Exception:
        pass
        
    return False
