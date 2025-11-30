1. Run model and test baseline performance
    - correctness for small sample of questions
    - ensure validity of hidden state outputs for 10 questions
2. Create data loader/logger and rest of scaffolding for data inputs (for topic probe)
3. Setup topic probe
4. (Hamza) use prompt template from Hailey and run through 50 questions
    - look at hidden state outputs for 10 questions

data loader → features → logistic regression → layer sweep (every 4th layer).
DoD: Plot of probe accuracy vs. layer; best layer identified.

----
1. Models to test/run
    DeepSeekMath‑7B,
    Mathstral‑7B,
