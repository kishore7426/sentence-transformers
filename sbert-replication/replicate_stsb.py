from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from scipy.stats import spearmanr

print("Loading STS Benchmark...")
dataset = load_dataset("sentence-transformers/stsb", split="test")

MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"

print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

sentences1 = dataset["sentence1"]
sentences2 = dataset["sentence2"]
gold_scores = dataset["score"]

print("Encoding sentence pairs...")

emb1 = model.encode(
    sentences1,
    batch_size=32,
    show_progress_bar=True
)

emb2 = model.encode(
    sentences2,
    batch_size=32,
    show_progress_bar=True
)

predicted_scores = []

for a, b in zip(emb1, emb2):
    predicted_scores.append(cos_sim(a, b).item())

correlation = spearmanr(predicted_scores, gold_scores)

print("\nNumber of test pairs:", len(gold_scores))
print("Spearman correlation:", correlation.statistic)
print("Spearman x 100:", correlation.statistic * 100)

paper_score = 77.03
our_score = correlation.statistic * 100
difference = abs(paper_score - our_score)

print("\n--- Replication Comparison ---")
print(f"Paper SBERT-NLI-base STSb: {paper_score:.2f}")
print(f"Reproduced STSb:           {our_score:.2f}")
print(f"Absolute difference:        {difference:.2f}")
