try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    # If math-verify is not available (e.g. local env), we'll rely on fallbacks.
    # Define dummy functions/classes to prevent NameErrors if used without checking.
    def parse(*args, **kwargs): return []
    def verify(*args, **kwargs): return False
    class LatexExtractionConfig: pass
    class ExprExtractionConfig: pass

def extract_answer(text):
    """
    Extracts the answer using math-verify's parser.
    We try to parse it into a format that math-verify can use for comparison.
    However, math-verify's verify() function takes the raw model output and the gold answer.
    So we might not need to explicitly 'extract' a string for the user unless we want to show it.
    
    For display purposes, we can try to find the boxed content.
    """
    # math-verify parses into a list of candidates.
    # We can use LatexExtractionConfig to find \boxed{} content.
    candidates = parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    if candidates:
        return str(candidates[-1]) # Return the last candidate as the "extracted" answer
    return text.strip()

def is_equivalent(model_ans, ground_truth):
    """
    Checks if model_ans is equivalent to ground_truth using math-verify.
    Args:
        model_ans (str): The raw model output (or extracted text).
        ground_truth (str): The gold answer.
    """
    # 1. Direct String Equality (Normalized)
    # This handles cases like "-2" == "-2" or "0" == "0" instantly.
    norm_ans = model_ans.strip().lower()
    norm_gt = ground_truth.strip().lower()
    if norm_ans == norm_gt:
        return True
        
    # 2. Basic Numeric Equality
    # Tries to parse both as floats. Handles "0" == "0.0" or "1" == "1.00"
    try:
        if float(norm_ans) == float(norm_gt):
            return True
    except ValueError:
        pass

    # 3. Math-Verify (The Heavy Lifter)
    try:
        if verify(ground_truth, model_ans):
            return True
            
        # Fallback: Try removing spaces. 
        # math-verify sometimes struggles with spacing differences in LaTeX.
        # e.g. "1+\sqrt{2}" vs "1 + \sqrt{2}"
        gt_no_space = ground_truth.replace(" ", "")
        ans_no_space = model_ans.replace(" ", "")
        if verify(gt_no_space, ans_no_space):
            return True
            
    except Exception as e:
        # If math-verify isn't installed or fails, we rely on the checks above.
        # In the local environment where math-verify is missing, this print might be noisy,
        # but in Colab it should be fine.
        # print(f"Error in math-verify: {e}") 
        pass
        
    return False

def check_truncation(text, is_truncated_flag=False):
    """
    Checks if the text appears truncated.
    Args:
        text (str): The generated text.
        is_truncated_flag (bool): The flag returned by the generation function.
    Returns:
        bool: True if truncated.
    """
    if is_truncated_flag:
        return True
        
    # Heuristic check: Does it end with punctuation or a closing brace?
    # This is less reliable than the flag, but useful if flag isn't available.
    text = text.strip()
    if not text:
        return True
        
    # Common endings for math proofs
    if text.endswith(".") or text.endswith("}") or text.endswith("]") or text.endswith(")"):
        return False
        
    # If it ends with a number or variable, it might be okay, but usually there's a period.
    # Let's rely mostly on the flag.
    return False
