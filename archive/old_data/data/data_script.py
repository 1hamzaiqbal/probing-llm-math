# This script is used to generate 75 problems for the model. 5 problems from
# each level of difficulty for each of the 3 topics: Algebra, Number Theory,
# and Counting & Probability. These are populated into problems.csv.

from datasets import load_dataset
import pandas as pd

dataset = load_dataset("qwedsacf/competition_math", split="train")

topics = ["Algebra", "Number Theory", "Counting & Probability"]
levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]

problems = []

for topic in topics:
    for level in levels:
        subset = dataset.filter(lambda x: x['type'] == topic and x['level'] == level)
        examples = subset.select(range(min(5, len(subset))))
        for ex in examples:
            problems.append({
                "topic": topic,
                "difficulty": level,
                "question": ex["problem"],
                "answer": ex["solution"]
            })

df = pd.DataFrame(problems)
df.insert(0, "id", range(1, len(df) + 1))

df.to_csv("problems.csv", index=False)

print("✅ Saved", len(df), "problems to problems.csv")

