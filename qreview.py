"""
Interactive review script for generated GPQA-style questions.
Navigate through questions and mark them as verified, rejected, or needs editing.
"""

import json
import sys
import os
from datetime import datetime

def load_batch(filepath: str) -> list[dict]:
    """Load a batch file and extract valid questions."""
    with open(filepath) as f:
        batch = json.load(f)
    
    # Extract only successful generations
    questions = []
    for i, item in enumerate(batch):
        if item.get("success"):
            q = item["data"]
            q["_index"] = i
            q["_topic"] = item.get("topic", "unknown")
            q["_style"] = item.get("style", "unknown")
            questions.append(q)
    
    return questions


def display_question(q: dict, num: int, total: int):
    """Display a single question for review."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 70)
    print(f"  QUESTION {num}/{total}  |  Topic: {q['_topic'][:40]}")
    print("=" * 70)
    
    print(f"\n{q['question']}\n")
    
    print("-" * 70)
    print(f"CORRECT ANSWER ({q.get('correct_letter', '?')}):")
    print(f"  {q['correct_answer'][:500]}")
    if len(q['correct_answer']) > 500:
        print("  [truncated...]")
    
    print("-" * 70)
    print("DISTRACTORS:")
    for i in range(1, 5):
        key = f'incorrect_{i}'
        letter_key = f'incorrect_{i}_letter'
        ans = q.get(key, "")
        letter = q.get(letter_key, "?")
        if ans:
            print(f"  {letter}) {ans[:200]}{'...' if len(ans) > 200 else ''}")
    
    print("-" * 70)
    print("REASONING (from Kimi):")
    thinking = q.get('thinking', '')[:600]
    print(f"  {thinking}{'...' if len(q.get('thinking', '')) > 600 else ''}")
    
    print("=" * 70)


def display_full(q: dict):
    """Display full question without truncation."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 70)
    print("  FULL QUESTION VIEW")
    print("=" * 70)
    
    print(f"\nQUESTION:\n{q['question']}\n")
    
    print("-" * 70)
    print(f"CORRECT ANSWER ({q.get('correct_letter', '?')}):\n{q['correct_answer']}\n")
    
    print("-" * 70)
    print("DISTRACTORS:")
    for i in range(1, 5):
        key = f'incorrect_{i}'
        letter_key = f'incorrect_{i}_letter'
        ans = q.get(key, "")
        letter = q.get(letter_key, "?")
        if ans:
            print(f"\n{letter}) {ans}")
    
    print("-" * 70)
    print(f"REASONING:\n{q.get('thinking', '')}")
    
    print("=" * 70)
    input("\nPress Enter to go back...")


def review_session(filepath: str):
    """Run an interactive review session."""
    
    questions = load_batch(filepath)
    
    if not questions:
        print("No valid questions found in batch.")
        return
    
    print(f"Loaded {len(questions)} questions from {filepath}")
    
    # Track review status
    reviews = {}
    for q in questions:
        reviews[q['_index']] = {"status": "pending", "notes": ""}
    
    # Review state
    current = 0
    
    while True:
        q = questions[current]
        display_question(q, current + 1, len(questions))
        
        # Show current status
        status = reviews[q['_index']]['status']
        status_display = {
            "pending": "⏳ PENDING",
            "verified": "✅ VERIFIED", 
            "rejected": "❌ REJECTED",
            "edit": "✏️  NEEDS EDIT"
        }
        print(f"\nStatus: {status_display.get(status, status)}")
        
        # Show commands
        print("\nCommands:")
        print("  [v] Verify   [r] Reject   [e] Needs edit   [n] Add note")
        print("  [f] Full view   [j] Next   [k] Previous   [g] Go to #")
        print("  [s] Summary   [w] Save & quit   [q] Quit without saving")
        
        cmd = input("\n> ").strip().lower()
        
        if cmd == 'v':
            reviews[q['_index']]['status'] = 'verified'
            print("Marked as VERIFIED")
            if current < len(questions) - 1:
                current += 1
        
        elif cmd == 'r':
            reason = input("Rejection reason (optional): ").strip()
            reviews[q['_index']]['status'] = 'rejected'
            reviews[q['_index']]['notes'] = reason
            print("Marked as REJECTED")
            if current < len(questions) - 1:
                current += 1
        
        elif cmd == 'e':
            note = input("What needs editing? ").strip()
            reviews[q['_index']]['status'] = 'edit'
            reviews[q['_index']]['notes'] = note
            print("Marked as NEEDS EDIT")
            if current < len(questions) - 1:
                current += 1
        
        elif cmd == 'n':
            note = input("Note: ").strip()
            reviews[q['_index']]['notes'] = note
            print("Note added")
        
        elif cmd == 'f':
            display_full(q)
        
        elif cmd == 'j':
            if current < len(questions) - 1:
                current += 1
            else:
                print("Already at last question")
                input("Press Enter...")
        
        elif cmd == 'k':
            if current > 0:
                current -= 1
            else:
                print("Already at first question")
                input("Press Enter...")
        
        elif cmd == 'g':
            try:
                num = int(input("Go to question #: "))
                if 1 <= num <= len(questions):
                    current = num - 1
                else:
                    print(f"Invalid. Enter 1-{len(questions)}")
                    input("Press Enter...")
            except ValueError:
                print("Invalid number")
                input("Press Enter...")
        
        elif cmd == 's':
            os.system('clear' if os.name == 'posix' else 'cls')
            print("=" * 70)
            print("  REVIEW SUMMARY")
            print("=" * 70)
            
            counts = {"pending": 0, "verified": 0, "rejected": 0, "edit": 0}
            for idx, rev in reviews.items():
                counts[rev['status']] += 1
            
            print(f"\n  ✅ Verified:    {counts['verified']}")
            print(f"  ❌ Rejected:    {counts['rejected']}")
            print(f"  ✏️  Needs edit:  {counts['edit']}")
            print(f"  ⏳ Pending:     {counts['pending']}")
            print(f"\n  Total: {len(questions)}")
            
            # Show rejected/edit items
            print("\n" + "-" * 70)
            print("Rejected/Needs Edit:")
            for i, q in enumerate(questions):
                rev = reviews[q['_index']]
                if rev['status'] in ['rejected', 'edit']:
                    print(f"  #{i+1} [{rev['status']}]: {rev['notes'][:50]}")
            
            print("=" * 70)
            input("\nPress Enter to continue...")
        
        elif cmd == 'w':
            save_reviews(filepath, questions, reviews)
            break
        
        elif cmd == 'q':
            confirm = input("Quit without saving? (y/n): ")
            if confirm.lower() == 'y':
                break


def save_reviews(original_path: str, questions: list, reviews: dict):
    """Save review results to separate files."""
    
    # Create output directories
    os.makedirs("data/verified", exist_ok=True)
    os.makedirs("data/rejected", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    verified = []
    rejected = []
    
    for q in questions:
        rev = reviews[q['_index']]
        
        # Clean up internal keys
        clean_q = {k: v for k, v in q.items() if not k.startswith('_')}
        clean_q['topic'] = q['_topic']
        clean_q['review_notes'] = rev['notes']
        
        if rev['status'] == 'verified':
            verified.append(clean_q)
        elif rev['status'] in ['rejected', 'edit']:
            clean_q['rejection_reason'] = rev['notes']
            clean_q['review_status'] = rev['status']
            rejected.append(clean_q)
    
    # Save verified
    if verified:
        outpath = f"data/verified/reviewed_{timestamp}.json"
        with open(outpath, 'w') as f:
            json.dump(verified, f, indent=2)
        print(f"Saved {len(verified)} verified questions to {outpath}")
    
    # Save rejected
    if rejected:
        outpath = f"data/rejected/rejected_{timestamp}.json"
        with open(outpath, 'w') as f:
            json.dump(rejected, f, indent=2)
        print(f"Saved {len(rejected)} rejected questions to {outpath}")
    
    # Summary
    pending = sum(1 for r in reviews.values() if r['status'] == 'pending')
    if pending:
        print(f"\nNote: {pending} questions still pending review")


def main():
    if len(sys.argv) < 2:
        # Find most recent batch file
        batch_dir = "data/raw"
        if os.path.exists(batch_dir):
            files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.json')])
            if files:
                default = os.path.join(batch_dir, files[-1])
                print(f"No file specified. Use most recent batch? ({default})")
                confirm = input("(y/n): ").strip().lower()
                if confirm == 'y':
                    review_session(default)
                    return
        
        print("Usage: python review.py <batch_file.json>")
        print("   or: python review.py  (to use most recent batch)")
        return
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    review_session(filepath)


if __name__ == "__main__":
    main()