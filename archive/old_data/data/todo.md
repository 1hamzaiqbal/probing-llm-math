1. Build a 150‑problem stratified slice from MATH (3 topics: algebra, number theory, combinatorics; difficulty 1–5).
    ex. problems.csv (id, topic, difficulty, question, answer).
2. Create prompt and answer prompt templates --> standardized format for all questions
3. Run initial eval: 5 samples/problem (Hamza needs to get the model running for this)
    - correctness per attempt
    - aggregated accuracy
    - Topic×difficulty accuracy heatmap + table
----

Dataset I think we go with:
https://huggingface.co/datasets/qwedsacf/competition_math/viewer/default/train?views%5B%5D=train&row=5
