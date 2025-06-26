import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from detoxify import Detoxify
from pathlib import Path
from tqdm import tqdm

# CONFIG: Set your dataset filename (just filename, not full path)
filename = "gemini_dataset_v1.csv"
dataset_path = Path("data/raw") / filename
output_path = Path("data/raw") / filename.replace(".csv", "_scored.csv")
graph_dir = Path("data/graphs")
graph_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)

# Load Detoxify model
model = Detoxify('original')

# Score posts (if not already done)
if 'toxicity_score' not in df.columns:
    tqdm.pandas(desc="Scoring posts")
    df['toxicity_score'] = df['post'].progress_apply(lambda x: model.predict(x)['toxicity'])
    df.to_csv(output_path, index=False)

sns.set(style="whitegrid")

# 1. Average Toxicity per Label
plt.figure(figsize=(8, 5))
avg_tox = df.groupby("label")["toxicity_score"].mean().sort_values()
sns.barplot(x=avg_tox.index, y=avg_tox.values)
plt.title("🔬 Average Detoxify Toxicity Score by Label")
plt.ylabel("Average Toxicity")
plt.xlabel("Post Label")
plt.tight_layout()
plt.savefig(graph_dir / "toxicity_avg_per_label.png")
plt.close()

# 2. OVERLAP: KDE Distribution for all labels on one plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="toxicity_score", hue="label", common_norm=False, fill=True, alpha=0.4)
plt.title("📈 Toxicity Score KDE Overlap by Label")
plt.xlabel("Toxicity Score")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(graph_dir / "toxicity_kde_overlap_by_label.png")
plt.close()

# 3. KDE Distribution for each label separately
for label in df['label'].unique():
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df[df['label'] == label],
        x="toxicity_score",
        fill=True,
        alpha=0.5,
        color="C0"
    )
    plt.title(f"Toxicity Score KDE Distribution: {label}")
    plt.xlabel("Toxicity Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(graph_dir / f"toxicity_kde_{label}.png")
    plt.close()

# 4. Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="label", y="toxicity_score")
plt.title("📦 Toxicity Score Box Plot by Label")
plt.ylabel("Toxicity Score")
plt.xlabel("Label")
plt.tight_layout()
plt.savefig(graph_dir / "toxicity_boxplot_by_label.png")
plt.close()

print("Done! All plots saved in", graph_dir)
