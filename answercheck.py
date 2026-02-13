"""Quick script to check answer distribution in a batch file."""
import json
import sys
from collections import Counter

def check_distribution(filepath):
    with open(filepath) as f:
        batch = json.load(f)
    
    letters = []
    for item in batch:
        if item.get("success"):
            letter = item.get("data", {}).get("correct_letter")
            if letter:
                letters.append(letter)
    
    total = len(letters)
    counts = Counter(letters)
    
    print(f"Answer distribution ({total} valid questions):\n")
    for letter in ['A', 'B', 'C', 'D', 'E']:
        count = counts.get(letter, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 5)
        print(f"  {letter}: {count:3d} ({pct:5.1f}%) {bar}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent batch
        import os
        batch_dir = "data/raw"
        if os.path.exists(batch_dir):
            files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.json')])
            if files:
                filepath = os.path.join(batch_dir, files[-1])
                print(f"Using: {filepath}\n")
                check_distribution(filepath)
            else:
                print("No batch files found")
        else:
            print("Usage: python check_answers.py <batch_file.json>")
    else:
        check_distribution(sys.argv[1])