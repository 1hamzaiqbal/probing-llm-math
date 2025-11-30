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


def extract_final_value(text):
    """
    Extract the final value from a derivation chain like "... = D" or "... = 42".
    Returns the rightmost value after the last equals sign.
    """
    text = text.strip()
    
    # Find the last equals sign and extract what follows
    # This handles chains like "a = b = c = 42"
    last_eq_pos = text.rfind('=')
    if last_eq_pos != -1:
        final = text[last_eq_pos + 1:].strip()
        # Remove trailing period if present
        final = final.rstrip('.')
        # Make sure it's not too long (a derivation, not a simple value)
        if len(final) < 50 and len(final) > 0:
            return final
    
    return None


def extract_answer(text):
    """
    Extracts the answer from model output.
    Priority: \boxed{} > final value from derivation > math-verify parser > raw text
    """
    # Try boxed extraction first
    boxed = extract_boxed(text)
    if boxed:
        return boxed.strip()
    
    # Try extracting final value from derivation (e.g., "... = D")
    final = extract_final_value(text)
    if final and len(final) < len(text):
        return final
    
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
    
    # Remove LaTeX spacing commands: \! \, \: \; \quad \qquad
    s = re.sub(r'\\[!,;:]', '', s)
    s = re.sub(r'\\q?quad', '', s)
    
    # Normalize whitespace (but preserve some structure)
    s = re.sub(r'\s+', ' ', s)
    
    # Normalize fraction commands: \dfrac -> \frac
    s = s.replace('\\dfrac', '\\frac')
    
    # Normalize shorthand fractions: \frac12 -> \frac{1}{2}, \frac1{2} -> \frac{1}{2}
    # Handle \fracXY where X and Y are single digits
    s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
    # Handle \fracX{Y} or \frac{X}Y
    s = re.sub(r'\\frac(\d)\{', r'\\frac{\1}{', s)
    s = re.sub(r'\}(\d)(?![0-9])', r'}{\1}', s)
    
    # Normalize shorthand sqrt: \sqrt6 -> \sqrt{6}, \sqrtX -> \sqrt{X}
    s = re.sub(r'\\sqrt(\d)(?!\d)', r'\\sqrt{\1}', s)
    s = re.sub(r'\\sqrt([a-zA-Z])(?![a-zA-Z{])', r'\\sqrt{\1}', s)
    
    # Normalize common variants
    s = s.replace('\\left(', '(').replace('\\right)', ')')
    s = s.replace('\\left[', '[').replace('\\right]', ']')
    s = s.replace('\\left{', '{').replace('\\right}', '}')
    s = s.replace('\\cdot', '*')
    s = s.replace('\\times', '*')
    
    # Normalize base notation: 4210_{5} -> 4210_5, 4210_{7} -> 4210_7
    s = re.sub(r'_\{(\d+)\}', r'_\1', s)
    
    # Strip units from answers: "575\text{ students}" -> "575"
    # Common units: students, multiples, square feet, feet, meters, etc.
    s = re.sub(r'\\text\{\s*(students?|multiples?|square feet|feet|meters?|units?|dollars?|cents?|people|items?|ways?|hours?|minutes?|seconds?|days?|years?|inches?|cm|mm|km|miles?|pounds?|kg|grams?|liters?|gallons?)\s*\}', '', s, flags=re.IGNORECASE)
    
    # Strip degree symbols: 840^\circ or 840° -> 840
    s = re.sub(r'\^\\circ', '', s)
    s = s.replace('°', '')
    
    # Handle \text{} wrapper for single letters/values: \text{A} -> A, \text{(E)} -> E
    # First, handle \text{(X)} -> X (multiple choice with parens)
    s = re.sub(r'\\text\{\(([A-Za-z])\)\}', r'\1', s)
    # Then, handle \text{X} -> X (plain single character)
    s = re.sub(r'\\text\{([A-Za-z0-9])\}', r'\1', s)
    
    return s.strip()


def latex_to_sympy_string(expr_str):
    """
    Convert LaTeX notation to a string that SymPy can parse with sympify.
    """
    s = expr_str.strip()
    
    # Normalize \dfrac to \frac
    s = s.replace('\\dfrac', '\\frac')
    
    # Handle space-separated \frac: \frac 59 -> \frac{5}{9}
    s = re.sub(r'\\frac\s+(\d)\s*(\d)', r'\\frac{\1}{\2}', s)
    # Also handle single args: \frac 5{9} or \frac{5} 9
    s = re.sub(r'\\frac\s*\{([^{}]+)\}\s*(\d)', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\frac\s+(\d)\s*\{([^{}]+)\}', r'\\frac{\1}{\2}', s)
    
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
    
    # Handle implicit multiplication:
    # - Number followed by parenthesis: 3(x) -> 3*(x), -3(x) -> -3*(x)
    # - Closing paren followed by opening paren: )(  -> )*(
    # - Number followed by variable: 2x -> 2*x, 3y -> 3*y
    s = re.sub(r'(\d)\(', r'\1*(', s)  # 3( -> 3*(
    s = re.sub(r'\)\(', r')*(', s)     # )( -> )*(
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 2x -> 2*x
    
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


def extract_tuple_elements(s):
    """
    Extract elements from a coordinate/tuple like (1, 2) or (1, \frac{9}{2}).
    Returns a list of element strings, or None if not a tuple.
    """
    s = s.strip()
    
    # Remove \left and \right modifiers
    s = s.replace('\\left(', '(').replace('\\right)', ')')
    s = s.replace('\\left[', '[').replace('\\right]', ']')
    s = s.strip()
    
    # Must start with ( and end with )
    if not (s.startswith('(') and s.endswith(')')):
        return None
    
    # Check it's not a matrix
    if 'matrix' in s.lower():
        return None
    
    content = s[1:-1].strip()
    
    # Split by comma, but be careful with nested structures
    elements = []
    depth = 0
    current = ""
    
    for char in content:
        if char in '({[':
            depth += 1
            current += char
        elif char in ')}]':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            elements.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        elements.append(current.strip())
    
    # Should have at least 2 elements for a valid tuple
    if len(elements) >= 2:
        return elements
    
    return None


def compare_tuple_elements(tup1, tup2):
    """
    Compare two tuples element-wise using symbolic comparison.
    """
    if tup1 is None or tup2 is None:
        return False
    
    if len(tup1) != len(tup2):
        return False
    
    for elem1, elem2 in zip(tup1, tup2):
        if not sympy_equivalent(elem1, elem2):
            return False
    
    return True


def normalize_number_string(s):
    r"""
    Normalize a number string by removing thousands separators and LaTeX formatting.
    E.g., "90,900,909" -> "90900909"
    E.g., "9,\!240" -> "9240"
    """
    s = s.strip()
    # Remove LaTeX spacing commands within numbers
    s = re.sub(r'\\[!,;:]', '', s)
    # Remove commas used as thousands separators
    # But be careful: (1, 2) has commas that aren't thousands separators
    # Only remove if it looks like a plain number with commas
    if re.match(r'^-?[\d,]+\.?\d*$', s):
        return s.replace(',', '')
    return s


def extract_set_elements(s):
    r"""
    Extract elements from a set like {1, 2, 3} or \{1, 2, 3\}.
    Returns a sorted list of element strings, or None if not a set.
    """
    s = s.strip()
    
    # Handle LaTeX set notation
    s = s.replace('\\{', '{').replace('\\}', '}')
    
    # Must start with { and end with }
    if not (s.startswith('{') and s.endswith('}')):
        return None
    
    content = s[1:-1].strip()
    
    # Split by comma
    elements = [elem.strip() for elem in content.split(',')]
    
    if len(elements) >= 1:
        return sorted(elements)
    
    return None


def compare_sets(set1, set2):
    """
    Compare two sets element-wise (order doesn't matter).
    """
    if set1 is None or set2 is None:
        return False
    
    if len(set1) != len(set2):
        return False
    
    # For sets, we need to find a matching for each element
    # Try to match each element from set1 to set2
    used = [False] * len(set2)
    
    for elem1 in set1:
        found = False
        for j, elem2 in enumerate(set2):
            if not used[j] and sympy_equivalent(elem1, elem2):
                used[j] = True
                found = True
                break
        if not found:
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
    
    # 0. If model answer contains '=' and is longer, try to extract final value
    # This handles cases like long derivations ending with "= D" where GT is just "D"
    # or chains like "a = b = c = 42" where GT is "42"
    if '=' in model_ans and len(model_ans) > len(ground_truth) + 5:
        final_value = extract_final_value(model_ans)
        if final_value:
            # Recursively check if the extracted value matches
            if is_equivalent(final_value, ground_truth):
                return True
    
    # 0.5. Handle variable prefixes: "x = 3" vs "3", "b=4" vs "4"
    # Strip common variable prefixes from both sides
    var_prefixes = [r'^[a-zA-Z]\s*=\s*', r'^[a-zA-Z]_?\d?\s*=\s*']
    ans_stripped = model_ans
    gt_stripped = ground_truth
    for pattern in var_prefixes:
        ans_stripped = re.sub(pattern, '', ans_stripped)
        gt_stripped = re.sub(pattern, '', gt_stripped)
    
    # If stripping changed something, recursively check
    if ans_stripped != model_ans or gt_stripped != ground_truth:
        if is_equivalent(ans_stripped, gt_stripped):
            return True
    
    # 1. Direct String Equality (Normalized)
    norm_ans = normalize_latex(model_ans).lower()
    norm_gt = normalize_latex(ground_truth).lower()
    if norm_ans == norm_gt:
        return True
    
    # Remove all whitespace for comparison
    if norm_ans.replace(' ', '') == norm_gt.replace(' ', ''):
        return True
    
    # 1.5. Strip outer parentheses: (4x - 7) vs 4x - 7
    def strip_outer_parens(s):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            # Check if the parens are balanced (not like "(a) + (b)")
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(s) - 1:
                    return s  # Parens aren't wrapping the whole thing
            return s[1:-1]
        return s
    
    norm_ans_no_parens = strip_outer_parens(norm_ans.replace(' ', ''))
    norm_gt_no_parens = strip_outer_parens(norm_gt.replace(' ', ''))
    if norm_ans_no_parens == norm_gt_no_parens:
        return True
    
    # 2. Normalize numbers with thousands separators (e.g., "90,900,909" -> "90900909")
    norm_ans_num = normalize_number_string(model_ans)
    norm_gt_num = normalize_number_string(ground_truth)
    if norm_ans_num != model_ans or norm_gt_num != ground_truth:
        # At least one had commas, try comparing normalized versions
        if norm_ans_num == norm_gt_num:
            return True
        try:
            if float(norm_ans_num) == float(norm_gt_num):
                return True
        except (ValueError, TypeError):
            pass
        
    # 3. Basic Numeric Equality
    try:
        if float(model_ans) == float(ground_truth):
            return True
    except (ValueError, TypeError):
        pass

    # 4. Check if both are tuples/coordinates and compare element-wise
    # e.g., (1, \frac{9}{2}) vs (1, 4.5)
    tup1 = extract_tuple_elements(model_ans)
    tup2 = extract_tuple_elements(ground_truth)
    if tup1 is not None and tup2 is not None:
        if compare_tuple_elements(tup1, tup2):
            return True

    # 5. Check if both are sets and compare element-wise (order independent)
    # e.g., {1, 2, 3} vs {3, 1, 2}
    set1 = extract_set_elements(model_ans)
    set2 = extract_set_elements(ground_truth)
    if set1 is not None and set2 is not None:
        if compare_sets(set1, set2):
            return True

    # 6. Check if both are matrices and compare element-wise
    if 'matrix' in model_ans.lower() or 'matrix' in ground_truth.lower():
        mat1 = extract_matrix_elements(model_ans)
        mat2 = extract_matrix_elements(ground_truth)
        if mat1 is not None and mat2 is not None:
            if compare_matrix_elements(mat1, mat2):
                return True

    # 7. Math-Verify (if available)
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

    # 8. SymPy Symbolic Comparison (handles fractions, radicals, complex numbers)
    if sympy_equivalent(model_ans, ground_truth):
        return True
    
    # 9. Special case: Complex numbers with different ordering (i/5 vs (1/5)i)
    try:
        ans_normalized = model_ans.replace('\\frac{i}', 'i*\\frac{1}')
        gt_normalized = ground_truth.replace('\\frac{i}', 'i*\\frac{1}')
        if sympy_equivalent(ans_normalized, gt_normalized):
            return True
    except Exception:
        pass
    
    # 10. Interval notation comparison (e.g., [1, \infty) vs [1, inf))
    # For now, rely on string matching after normalization
    # Intervals with infinity are tricky for symbolic comparison
    if ('\\infty' in model_ans or '\\infty' in ground_truth or 
        'infty' in model_ans or 'infty' in ground_truth or
        'inf' in model_ans.lower() or 'inf' in ground_truth.lower()):
        # Normalize infinity representations
        ans_inf = model_ans.replace('\\infty', 'oo').replace('infty', 'oo').lower()
        gt_inf = ground_truth.replace('\\infty', 'oo').replace('infty', 'oo').lower()
        ans_inf = re.sub(r'\s+', '', ans_inf)
        gt_inf = re.sub(r'\s+', '', gt_inf)
        if ans_inf == gt_inf:
            return True
        
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
