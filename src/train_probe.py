import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def train_probes(data_file="probe_data.pt", output_dir="probe_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading data from {data_file}...")
    data = torch.load(data_file)
    
    # Prepare data
    # data is a list of dicts
    # We want to aggregate activations per layer
    
    num_layers = data[0]['prompt_activations'].shape[0]
    hidden_dim = data[0]['prompt_activations'].shape[1]
    
    labels = np.array([1 if item['is_correct'] else 0 for item in data])
    
    print(f"Total samples: {len(labels)}")
    print(f"Class balance: {sum(labels)} correct, {len(labels)-sum(labels)} incorrect")
    
    if len(set(labels)) < 2:
        print("Error: Need both correct and incorrect samples to train a classifier.")
        return

    # Store results
    results_prompt = []
    results_response = []
    
    print("Training probes per layer...")
    
    for layer_idx in range(num_layers):
        # Extract activations for this layer
        X_prompt = np.array([item['prompt_activations'][layer_idx].numpy() for item in data])
        X_response = np.array([item['response_activations'][layer_idx].numpy() for item in data])
        
        # Split train/test
        X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_prompt, labels, test_size=0.2, random_state=42)
        X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_response, labels, test_size=0.2, random_state=42)
        
        # Train Prompt Probe
        clf_p = LogisticRegression(max_iter=1000, solver='liblinear')
        clf_p.fit(X_p_train, y_p_train)
        y_p_pred = clf_p.predict(X_p_test)
        acc_p = accuracy_score(y_p_test, y_p_pred)
        f1_p = f1_score(y_p_test, y_p_pred)
        results_prompt.append(acc_p)
        
        # Train Response Probe
        clf_r = LogisticRegression(max_iter=1000, solver='liblinear')
        clf_r.fit(X_r_train, y_r_train)
        y_r_pred = clf_r.predict(X_r_test)
        acc_r = accuracy_score(y_r_test, y_r_pred)
        f1_r = f1_score(y_r_test, y_r_pred)
        results_response.append(acc_r)
        
        if layer_idx % 5 == 0:
            print(f"Layer {layer_idx}: Prompt Acc={acc_p:.2f}, Response Acc={acc_r:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results_prompt, label="Prompt (Last Token)")
    plt.plot(results_response, label="Response (Last Token)")
    plt.xlabel("Layer Index")
    plt.ylabel("Probing Accuracy")
    plt.title("Linear Probe Accuracy by Layer (Correctness Prediction)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/probe_accuracy.png")
    print(f"Saved accuracy plot to {output_dir}/probe_accuracy.png")
    
    # PCA Visualization for the best layer (usually middle-late)
    # Let's pick layer 20 as a heuristic representative
    target_layer = min(20, num_layers - 1)
    print(f"Generating PCA for layer {target_layer}...")
    
    X_viz = np.array([item['response_activations'][target_layer].numpy() for item in data])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_viz)
    
    plt.figure(figsize=(8, 8))
    colors = ['red' if l == 0 else 'blue' for l in labels]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of Hidden States (Layer {target_layer}) - Red: Incorrect, Blue: Correct")
    plt.savefig(f"{output_dir}/pca_layer_{target_layer}.png")
    print(f"Saved PCA plot to {output_dir}/pca_layer_{target_layer}.png")

if __name__ == "__main__":
    train_probes()
