from math_verify import verify

gold = "1+\\sqrt{2}"
pred = "1 + \\sqrt{2}"

print(f"Gold: '{gold}'")
print(f"Pred: '{pred}'")

try:
    result = verify(gold, pred)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
