import re
import sympy
from sympy.parsing.latex import parse_latex
from sympy import simplify, nsimplify, Rational, sqrt, I, Matrix, Abs

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    def parse(*args, **kwargs): return []
    def verify(*args, **kwargs): return False
    class LatexExtractionConfig: pass
    class ExprExtractionConfig: pass


def extract_boxed(text):
    """
    Extract content from \boxed{...}, handling nested braces.
    Returns the last boxed content found, or None if not found.
    """
    # Find all \boxed{ positions
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return None
    
    # Get the last match
    last_match = matches[-1]
    start = last_match.end()
    
    # Count braces to find matching close
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    
    if depth == 0:
        return text[start:pos-1]
    return None


def extract_answer(text):
    """
    Extracts the answer from model output.
    Priority: \boxed{} > math-verify parser > raw text
    """
    # Try boxed extraction first
    boxed = extract_boxed(text)
    if boxed:
        return boxed.strip()
    
    # Try math-verify parser
    if MATH_VERIFY_AVAILABLE:
        try:
            candidates = parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if candidates:
                return str(candidates[-1])
        except Exception:
            pass
    
    return text.strip()


def normalize_latex(s):
    """
    Normalize LaTeX string for comparison.
    """
    s = s.strip()
    # Remove display math delimiters
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^\\\[|\\\]$', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    # Normalize common variants
    s = s.replace('\\left(', '(').replace('\\right)', ')')
    s = s.replace('\\left[', '[').replace('\\right]', ']')
    s = s.replace('\\left{', '{').replace('\\right}', '}')
    s = s.replace('\\cdot', '*')
    s = s.replace('\\times', '*')
    return s.strip()


def latex_to_sympy_string(expr_str):
    """
    Convert LaTeX notation to a string that SymPy can parse with sympify.
    """
    s = expr_str.strip()
    
    # Handle \frac{a}{b} -> (a)/(b)
    # Need to handle nested braces properly
    while '\\frac' in s:
        match = re.search(r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', s)
        if match:
            num = match.group(1)
            den = match.group(2)
            # Recursively convert numerator and denominator
            num_conv = latex_to_sympy_string(num)
            den_conv = latex_to_sympy_string(den)
            s = s[:match.start()] + f'(({num_conv})/({den_conv}))' + s[match.end():]
        else:
            break
    
    # Handle \sqrt{x} -> sqrt(x)
    # Also handle implicit multiplication before sqrt: 2\sqrt{3} -> 2*sqrt(3)
    while '\\sqrt' in s:
        match = re.search(r'([\d\)])?\\sqrt\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', s)
        if match:
            prefix = match.group(1) or ''
            content = latex_to_sympy_string(match.group(2))
            if prefix:
                replacement = f'{prefix}*sqrt({content})'
            else:
                replacement = f'sqrt({content})'
            s = s[:match.start()] + replacement + s[match.end():]
        else:
            break
    
    # Handle remaining backslashes
    s = s.replace('\\cdot', '*')
    s = s.replace('\\times', '*')
    s = s.replace('\\pi', 'pi')
    s = s.replace('\\', '')
    
    # Handle implicit multiplication with i (imaginary unit)
    # Cases: 2i, )i, xi at end of expression
    # But be careful not to break things like 'pi', 'sin', etc.
    
    # First, protect 'pi' and other common math terms
    s = re.sub(r'\bpi\b', 'PI_PLACEHOLDER', s)
    
    # Handle i after a closing paren: )i -> )*I
    s = re.sub(r'\)i\b', ')*I', s)
    
    # Handle i after a number: 2i -> 2*I  
    s = re.sub(r'(\d)i\b', r'\1*I', s)
    
    # Handle standalone i or i at word boundary
    s = re.sub(r'\bi\b', 'I', s)
    
    # Restore pi
    s = s.replace('PI_PLACEHOLDER', 'pi')
    
    return s


def parse_latex_safe(expr_str):
    """
    Safely parse a LaTeX expression to SymPy, with multiple fallback strategies.
    """
    expr_str = normalize_latex(expr_str)
    
    # Strategy 1: Direct parse_latex (works for many standard expressions)
    try:
        result = parse_latex(expr_str)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 2: Convert LaTeX to SymPy-friendly string and sympify
    try:
        sympy_str = latex_to_sympy_string(expr_str)
        result = sympy.sympify(sympy_str, locals={'I': I, 'sqrt': sqrt, 'pi': sympy.pi})
        return result
    except Exception:
        pass
    
    # Strategy 3: Handle simple fractions like "2/5" directly
    if '/' in expr_str and '\\' not in expr_str:
        try:
            return sympy.sympify(expr_str)
        except Exception:
            pass
    
    # Strategy 4: Try sympify on minimally cleaned string
    try:
        cleaned = expr_str.replace('\\', '')
        cleaned = re.sub(r'frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', cleaned)
        cleaned = re.sub(r'sqrt\{([^}]*)\}', r'sqrt(\1)', cleaned)
        return sympy.sympify(cleaned)
    except Exception:
        pass
    
    return None


def extract_matrix_elements(s):
    """
    Extract elements from a pmatrix/bmatrix LaTeX expression.
    Returns a list of rows, where each row is a list of element strings.
    """
    # Find matrix content
    matrix_match = re.search(r'\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}', s, re.DOTALL)
    if not matrix_match:
        # Try simpler \begin{pmatrix} format or bare pmatrix
        matrix_match = re.search(r'\\[pb]matrix\s*(.*?)\\end[pb]matrix', s, re.DOTALL)
    if not matrix_match:
        return None
    
    content = matrix_match.group(1).strip()
    
    # Split by row separator \\
    rows = re.split(r'\\\\', content)
    
    result = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        # Split by & for columns
        elements = [elem.strip() for elem in row.split('&')]
        result.append(elements)
    
    return result


def compare_matrix_elements(mat1, mat2):
    """
    Compare two matrices element-wise using symbolic comparison.
    """
    if mat1 is None or mat2 is None:
        return False
    
    if len(mat1) != len(mat2):
        return False
    
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return False
        for elem1, elem2 in zip(row1, row2):
            if not sympy_equivalent(elem1, elem2):
                return False
    
    return True


def sympy_equivalent(expr1_str, expr2_str):
    """
    Check if two expressions are symbolically equivalent using SymPy.
    """
    expr1 = parse_latex_safe(expr1_str)
    expr2 = parse_latex_safe(expr2_str)
    
    if expr1 is None or expr2 is None:
        return False
    
    try:
        # Try direct equality first
        if expr1.equals(expr2):
            return True
    except Exception:
        pass
    
    try:
        # Try simplifying the difference
        diff = simplify(expr1 - expr2)
        if diff == 0:
            return True
    except Exception:
        pass
    
    try:
        # Try numerical evaluation for approximate equality
        val1 = complex(expr1.evalf())
        val2 = complex(expr2.evalf())
        if abs(val1 - val2) < 1e-9:
            return True
    except Exception:
        pass
    
    try:
        # Try nsimplify for rationalized forms
        simp1 = nsimplify(expr1)
        simp2 = nsimplify(expr2)
        if simp1.equals(simp2):
            return True
        if simplify(simp1 - simp2) == 0:
            return True
    except Exception:
        pass
    
    return False


def is_equivalent(model_ans, ground_truth):
    """
    Checks if model_ans is equivalent to ground_truth.
    Uses multiple strategies in order of computational cost.
    """
    # Normalize inputs
    model_ans = str(model_ans).strip()
    ground_truth = str(ground_truth).strip()
    
    # 1. Direct String Equality (Normalized)
    norm_ans = normalize_latex(model_ans).lower()
    norm_gt = normalize_latex(ground_truth).lower()
    if norm_ans == norm_gt:
        return True
    
    # Remove all whitespace for comparison
    if norm_ans.replace(' ', '') == norm_gt.replace(' ', ''):
        return True
        
    # 2. Basic Numeric Equality
    try:
        if float(model_ans) == float(ground_truth):
            return True
    except (ValueError, TypeError):
        pass

    # 3. Check if both are matrices and compare element-wise
    if 'matrix' in model_ans.lower() or 'matrix' in ground_truth.lower():
        mat1 = extract_matrix_elements(model_ans)
        mat2 = extract_matrix_elements(ground_truth)
        if mat1 is not None and mat2 is not None:
            if compare_matrix_elements(mat1, mat2):
                return True

    # 4. Math-Verify (if available)
    if MATH_VERIFY_AVAILABLE:
        try:
            if verify(ground_truth, model_ans):
                return True
        except Exception:
            pass
        
        # Try with spaces removed
        try:
            gt_no_space = ground_truth.replace(" ", "")
            ans_no_space = model_ans.replace(" ", "")
            if verify(gt_no_space, ans_no_space):
                return True
        except Exception:
            pass

    # 5. SymPy Symbolic Comparison (handles fractions, radicals, complex numbers)
    if sympy_equivalent(model_ans, ground_truth):
        return True
    
    # 6. Special case: Complex numbers with different ordering (i/5 vs (1/5)i)
    # Normalize complex number representations
    try:
        # Try to parse and compare as complex expressions
        ans_normalized = model_ans.replace('\\frac{i}', 'i*\\frac{1}')
        gt_normalized = ground_truth.replace('\\frac{i}', 'i*\\frac{1}')
        if sympy_equivalent(ans_normalized, gt_normalized):
            return True
    except Exception:
        pass
        
    return False


def check_truncation(text, is_truncated_flag=False):
    """
    Checks if the text appears truncated.
    """
    if is_truncated_flag:
        return True
        
    text = str(text).strip()
    if not text:
        return True
        
    # Common endings for complete math answers
    if text.endswith(('.', '}', ']', ')', '$')):
        return False
        
    return False


# Debugging helper
def debug_comparison(model_ans, ground_truth):
    """
    Debug why two answers might not be matching.
    Call this to see what's happening in the comparison pipeline.
    """
    print(f"Model answer: {repr(model_ans)}")
    print(f"Ground truth: {repr(ground_truth)}")
    print()
    
    norm_ans = normalize_latex(model_ans)
    norm_gt = normalize_latex(ground_truth)
    print(f"Normalized model: {repr(norm_ans)}")
    print(f"Normalized truth: {repr(norm_gt)}")
    print()
    
    expr1 = parse_latex_safe(model_ans)
    expr2 = parse_latex_safe(ground_truth)
    print(f"SymPy parsed model: {expr1}")
    print(f"SymPy parsed truth: {expr2}")
    print()
    
    if expr1 is not None and expr2 is not None:
        try:
            diff = simplify(expr1 - expr2)
            print(f"Difference simplified: {diff}")
        except Exception as e:
            print(f"Could not compute difference: {e}")
    
    # Check matrix
    mat1 = extract_matrix_elements(model_ans)
    mat2 = extract_matrix_elements(ground_truth)
    if mat1 or mat2:
        print(f"Matrix 1 elements: {mat1}")
        print(f"Matrix 2 elements: {mat2}")
    
    print()
    result = is_equivalent(model_ans, ground_truth)
    print(f"Final result: {result}")
    return result
