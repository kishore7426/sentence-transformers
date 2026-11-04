from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

print("Loading model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "A man is playing football.",
    "A person is playing soccer.",
    "The stock market fell today."
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)

similar = cos_sim(embeddings[0], embeddings[1]).item()
different = cos_sim(embeddings[0], embeddings[2]).item()

print("Football vs soccer:", similar)
print("Football vs stock market:", different)

if similar > different:
    print("SUCCESS: semantically similar sentences received a higher similarity score.")
else:
    print("Unexpected result.")
