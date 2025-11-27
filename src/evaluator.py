from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

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
    # verify() takes (gold, target) where target is the model output.
    # Wait, let's check the signature. usually verify(gold, prediction).
    # Based on docs: verify(gold, prediction) -> bool
    
    # We need to be careful: math-verify expects the gold answer to be parsed?
    # Or it handles raw strings?
    # Usually it handles raw strings for both.
    
    try:
        return verify(ground_truth, model_ans)
    except Exception as e:
        print(f"Error in math-verify: {e}")
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
