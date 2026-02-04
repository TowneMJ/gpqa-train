import sys
from datasets import load_dataset

gpqa_ext = load_dataset("Idavidrein/gpqa", "gpqa_extended")
bio = gpqa_ext["train"].filter(lambda x: x["High-level domain"] == "Biology")

n = int(sys.argv[1]) - 1
q = bio[n]

print(f"Q: {q['Question']}\n")
print(f"=== CORRECT ===\n{q['Correct Answer']}\n")
print(f"=== WRONG 1 ===\n{q['Incorrect Answer 1']}\n")
print(f"=== WRONG 2 ===\n{q['Incorrect Answer 2']}\n")
print(f"=== WRONG 3 ===\n{q['Incorrect Answer 3']}")