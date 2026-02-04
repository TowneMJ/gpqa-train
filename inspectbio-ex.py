from datasets import load_dataset

gpqa_ext = load_dataset("Idavidrein/gpqa", "gpqa_extended")
bio = gpqa_ext["train"].filter(lambda x: x["High-level domain"] == "Biology")

# Summary stats
print(f"Total: {len(bio)}")
print(f"\nSubdomain breakdown:")
from collections import Counter
print(Counter(bio["Subdomain"]))

# Look at a few questions (just the key fields)
print("\n" + "="*60)
for i in range(5):
    q = bio[i]
    print(f"\n--- Question {i+1} [{q['Subdomain']}] ---")
    print(f"Q: {q['Question'][:300]}...")
    print(f"\nCorrect: {q['Correct Answer'][:150]}...")
    print(f"\nWrong 1: {q['Incorrect Answer 1'][:150]}...")