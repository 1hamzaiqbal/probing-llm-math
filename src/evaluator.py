import re
from sympy import simplify, sympify
from sympy.parsing.sympy_parser import parse_expr

def extract_answer(text):
    """
    Extracts the final answer from the model output.
    Since we enforce "only the final answer", we can often just take the text.
    But sometimes models chatter. We'll look for the last number or simple expression.
    """
    text = text.strip()
    
    # Common patterns: "The answer is X", "X", "Boxed{X}" (if using other models)
    # For our strict prompt, it should be just X.
    # But let's try to be robust.
    
    # If it contains "####", take what's after (GSM8K style)
    if "####" in text:
        return text.split("####")[-1].strip()
    
    # If it's a single line, assume it's the answer
    lines = text.split('\n')
    if len(lines) == 1:
        return text
        
    # Otherwise, take the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
            
    return text

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
        # Parse expressions
        expr_model = parse_expr(model_ans)
        expr_gt = parse_expr(ground_truth)
        
        # Check if difference simplifies to 0
        diff = simplify(expr_model - expr_gt)
        if diff == 0:
            return True
    except Exception:
        pass
        
    return False
