from datasets import load_dataset

try:
    dataset = load_dataset("EleutherAI/hendrycks_math", "number_theory", split="test", trust_remote_code=True)
    print("Dataset loaded successfully.")
    print(f"Features: {dataset.features}")
    print(f"Number of samples: {len(dataset)}")
    print("Sample entry:")
    print(dataset[0])
    
    # Check for difficulty field
    if 'level' in dataset[0]:
        print("Difficulty levels found.")
        # Print unique levels if possible (might take time to iterate, just print first few)
        for i in range(5):
            print(f"Item {i} Level: {dataset[i].get('level')}")
except Exception as e:
    print(f"Error: {e}")
