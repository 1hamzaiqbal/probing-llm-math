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
    
    # 1. Look for \boxed{...}
    # This is the standard for math models.
    # We need to handle nested braces if possible, but for now simple regex is often enough for single numbers.
    # A better regex for balanced braces is hard, but let's try a recursive approach or a greedy one.
    
    # Simple greedy regex for \boxed{content} where content doesn't contain braces
    # This covers \boxed{4}, \boxed{2x}, etc.
    simple_boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if simple_boxed:
        return simple_boxed[-1].strip()
        
    # If nested, we might need a more complex parser. 
    # Let's try to find the last \boxed and match braces manually.
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

    # 2. Look for "The answer is X" patterns
    # ...
    
    # 3. GSM8K style "####"
    if "####" in text:
        return text.split("####")[-1].strip()
        
    # 4. Fallback: Last non-empty line, trying to strip LaTeX commands
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
        
    last_line = lines[-1]
    # Remove "Thus, ..." or "Therefore, ..."
    if last_line.lower().startswith("thus,") or last_line.lower().startswith("therefore,"):
        # Try to extract the math part
        # e.g. "Thus, the answer is 4." -> "4"
        pass
        
    return last_line

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
