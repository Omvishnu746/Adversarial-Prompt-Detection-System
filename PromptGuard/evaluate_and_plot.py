import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from inference_pipeline import analyze_prompt

# Load dataset
df = pd.read_csv("merged_dataset_new.csv")

# Ground truth label
df["true_label"] = df.apply(
    lambda row: 1 if row["jailbreak"] == 1 or row["prompt_injection"] == 1 else 0,
    axis=1
)

results = []

print("Running PromptGuard on dataset...")

for prompt in tqdm(df["Prompt"]):
    res = analyze_prompt(prompt)

    results.append({
        "prompt": prompt,
        "rule_score": res["rule_score"],
        "model_probability": res["model_probability"],
        "final_score": res["final_score"],
        "decision": res["decision"]
    })

results_df = pd.DataFrame(results)

# Convert decision to binary
results_df["predicted_label"] = results_df["decision"].apply(
    lambda x: 1 if x == "Block" else 0
)

# Merge results with ground truth
full_df = pd.concat([df, results_df], axis=1)

# -----------------------------
# 1. Prompt Category Distribution
# -----------------------------

categories = [
    "jailbreak",
    "role_manipulation",
    "prompt_injection",
    "indirect_injection",
    "obfuscation",
    "benign"
]

counts = [df[c].sum() for c in categories]

plt.figure()
plt.bar(categories, counts)
plt.xticks(rotation=45)
plt.title("Prompt Category Distribution")
plt.ylabel("Count")
plt.xlabel("Prompt Type")
plt.show()

# -----------------------------
# 2. Allowed vs Blocked
# -----------------------------

plt.figure()
results_df["decision"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Allowed vs Blocked Prompts")
plt.ylabel("")
plt.show()

# -----------------------------
# 3. Risk Score Distribution
# -----------------------------

plt.figure()
sns.histplot(results_df["final_score"], bins=30)
plt.title("Final Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 4. Model Probability Distribution
# -----------------------------

plt.figure()
sns.histplot(results_df["model_probability"], bins=30)
plt.title("Model Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 5. Rule Score Distribution
# -----------------------------

plt.figure()
sns.histplot(results_df["rule_score"], bins=10)
plt.title("Rule Engine Score Distribution")
plt.xlabel("Rule Score")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 6. Confusion Matrix
# -----------------------------

cm = confusion_matrix(full_df["true_label"], full_df["predicted_label"])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 7. Precision / Recall / F1
# -----------------------------

print("\nClassification Report:")
print(classification_report(full_df["true_label"], full_df["predicted_label"]))

# -----------------------------
# 8. Prompt Length vs Risk Score
# -----------------------------

full_df["prompt_length"] = full_df["Prompt"].apply(len)

plt.figure()
plt.scatter(full_df["prompt_length"], results_df["final_score"])
plt.title("Prompt Length vs Risk Score")
plt.xlabel("Prompt Length")
plt.ylabel("Risk Score")
plt.show()