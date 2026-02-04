from datasets import load_dataset
from collections import Counter

gpqa_ext = load_dataset("Idavidrein/gpqa", "gpqa_extended")
bio = gpqa_ext["train"].filter(lambda x: x["High-level domain"] == "Biology")

# Get first few words of each question
openers = []
for q in bio:
    words = q["Question"].split()[:5]
    openers.append(" ".join(words))

# Print all openers to see variety
print("=== Question openers ===\n")
for i, opener in enumerate(openers):
    print(f"{i+1}. {opener}...")

# Also look at first word distribution
first_words = [q["Question"].split()[0] for q in bio]
print("\n=== First word frequency ===")
print(Counter(first_words).most_common(15))