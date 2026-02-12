"""
Tiered review script for GPQA-style questions.
Implements multi-stage screening:
1. Human selects confident domains for direct review
2. Remaining questions go through Kimi self-check
3. Then Gemini evaluation (with adjudication context if Kimi disagreed)
4. Failed screens go to human review with AI notes
"""

import json
import sys
import os
import re
import random
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === API CLIENTS ===

# Kimi via NVIDIA NIMs
KIMI_CLIENT = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)
KIMI_MODEL = "moonshotai/kimi-k2.5"

# Gemini via Google AI Studio
GEMINI_CLIENT = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_MODEL = "gemini-3-flash-preview"


# === DATA LOADING ===

def load_batch(filepath: str) -> list[dict]:
    """Load a batch file and extract valid questions."""
    with open(filepath) as f:
        batch = json.load(f)
    
    questions = []
    for i, item in enumerate(batch):
        if item.get("success"):
            q = item["data"]
            q["_index"] = i
            q["_domain"] = item.get("domain", item.get("topic", "unknown"))
            q["_core_concept"] = item.get("core_concept", q.get("core_concept", ""))
            q["_style"] = item.get("style", "unknown")
            questions.append(q)
    
    return questions


def get_domains(questions: list[dict]) -> list[str]:
    """Extract unique domains from questions."""
    domains = set()
    for q in questions:
        domains.add(q["_domain"])
    return sorted(domains)


# === AI SCREENING FUNCTIONS ===

def call_kimi(prompt: str, max_tokens: int = 4096) -> dict:
    """Call Kimi API."""
    try:
        response = KIMI_CLIENT.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature to reduce rambling
            max_tokens=max_tokens
        )
        
        # Kimi in thinking mode may return content in different fields
        message = response.choices[0].message
        content = message.content
        
        # If content is None, check for reasoning_content
        if content is None:
            # Try to get reasoning_content from the message object
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            elif hasattr(message, 'reasoning') and message.reasoning:
                content = message.reasoning
            # Try accessing as dict
            try:
                msg_dict = message.model_dump() if hasattr(message, 'model_dump') else message.__dict__
                if not content and msg_dict.get('reasoning_content'):
                    content = msg_dict['reasoning_content']
                if not content and msg_dict.get('reasoning'):
                    content = msg_dict['reasoning']
            except:
                pass
        
        return {
            "success": True,
            "content": content
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_gemini(prompt: str, max_tokens: int = 4096, max_retries: int = 3) -> dict:
    """Call Gemini API with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = GEMINI_CLIENT.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )
            return {
                "success": True,
                "content": response.choices[0].message.content
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                print(f"Rate limited, waiting {wait_time}s...", end=" ", flush=True)
                time.sleep(wait_time)
                continue
            return {"success": False, "error": error_str}


def format_question_for_screening(q: dict) -> str:
    """Format a question for AI screening (no correct answer indicated)."""
    # Collect all options and shuffle them
    options = []
    
    correct = q.get("correct_answer", "")
    correct_letter = q.get("correct_letter", "A")
    
    # Build list of (original_letter, text) tuples
    options.append((correct_letter, correct))
    
    for i in range(1, 5):
        key = f"incorrect_{i}"
        letter_key = f"incorrect_{i}_letter"
        if q.get(key):
            options.append((q.get(letter_key, chr(65+i)), q[key]))
    
    # Shuffle options
    random.shuffle(options)
    
    # Reassign letters A-E
    letter_mapping = {}  # new_letter -> original_letter
    formatted_options = []
    for i, (orig_letter, text) in enumerate(options):
        new_letter = chr(65 + i)  # A, B, C, D, E
        letter_mapping[new_letter] = orig_letter
        formatted_options.append(f"{new_letter}) {text}")
    
    # Find which new letter corresponds to the correct answer
    correct_new_letter = None
    for new, orig in letter_mapping.items():
        if orig == correct_letter:
            correct_new_letter = new
            break
    
    question_text = f"""{q.get('question', '')}

{chr(10).join(formatted_options)}"""
    
    return question_text, correct_new_letter, letter_mapping


def kimi_screen(q: dict) -> dict:
    """
    Screen question with Kimi self-check.
    Returns dict with 'passed', 'kimi_answer', 'kimi_reasoning', 'error'.
    """
    question_text, correct_letter, letter_mapping = format_question_for_screening(q)
    
    prompt = f"""Answer this multiple choice question.

{question_text}

Respond with ONLY:
1. A brief explanation (4-5 sentences max)
2. Your final answer in the format: ANSWER: X

Keep your response under 200 words total."""

    response = call_kimi(prompt)
    
    if not response["success"]:
        return {
            "passed": None,
            "error": response["error"],
            "kimi_answer": None,
            "kimi_reasoning": None
        }
    
    content = response["content"]
    
    # Guard against None or non-string content
    if not content or not isinstance(content, str):
        return {
            "passed": None,
            "error": "Kimi returned empty or invalid response",
            "kimi_answer": None,
            "kimi_reasoning": f"Raw response type: {type(content)}, value: {repr(content)[:200]}"
        }
    
    # Parse answer - try multiple patterns
    kimi_answer = None
    answer_match = re.search(r'ANSWER:\s*([A-E])', content, re.IGNORECASE)
    if answer_match:
        kimi_answer = answer_match.group(1).upper()
    else:
        # Fallback: look for "The answer is X" or similar
        alt_match = re.search(r'(?:the answer is|I choose|I select|my answer is|final answer:?)\s*\(?([A-E])\)?', content, re.IGNORECASE)
        if alt_match:
            kimi_answer = alt_match.group(1).upper()
        else:
            # Last resort: look for standalone letter at end
            end_match = re.search(r'\b([A-E])\s*$', content.strip())
            if end_match:
                kimi_answer = end_match.group(1).upper()
    
    # Extract reasoning (everything before the answer pattern)
    reasoning = content
    if answer_match:
        reasoning = content[:answer_match.start()].strip()
    
    # Determine if passed
    passed = None
    if kimi_answer:
        passed = kimi_answer == correct_letter
    
    return {
        "passed": passed,
        "kimi_answer": kimi_answer,
        "kimi_answer_original": letter_mapping.get(kimi_answer, "?") if kimi_answer else None,
        "correct_letter_shuffled": correct_letter,
        "kimi_reasoning": reasoning,
        "error": None
    }


def gemini_screen(q: dict, kimi_result: dict = None) -> dict:
    """
    Screen question with Gemini.
    If kimi_result shows disagreement, use adjudication mode.
    Returns dict with 'passed', 'gemini_verdict', 'gemini_reasoning', 'error'.
    """
    question_text = q.get("question", "")
    correct_answer = q.get("correct_answer", "")
    correct_letter = q.get("correct_letter", "?")
    original_reasoning = q.get("thinking", "")
    
    # Build options display
    options_display = f"{correct_letter}) {correct_answer}\n"
    for i in range(1, 5):
        key = f"incorrect_{i}"
        letter_key = f"incorrect_{i}_letter"
        if q.get(key):
            options_display += f"{q.get(letter_key, '?')}) {q[key]}\n"
    
    # Check if we're in adjudication mode
    adjudication_mode = kimi_result and kimi_result.get("passed") == False and kimi_result.get("kimi_answer")
    
    if adjudication_mode:
        kimi_picked = kimi_result.get("kimi_answer_original", "?")
        kimi_reasoning = kimi_result.get("kimi_reasoning", "No reasoning provided")
        
        prompt = f"""Review this biology exam question. Be concise (under 300 words).

QUESTION:
{question_text}

OPTIONS:
{options_display}

STATED CORRECT ANSWER: {correct_letter}

STATED REASONING:
{original_reasoning}

---

CONTEXT: Another model selected {kimi_picked} instead of {correct_letter}.

---

First line must be: VERDICT: PASS or VERDICT: FAIL

Then briefly explain (2-3 paragraphs max):
- Is {correct_letter} unambiguously correct?
- Is {kimi_picked} defensible?
- Any logical flaws?"""

    else:
        prompt = f"""Review this biology exam question. Be concise (under 300 words).

QUESTION:
{question_text}

OPTIONS:
{options_display}

STATED CORRECT ANSWER: {correct_letter}

STATED REASONING:
{original_reasoning}

---

First line must be: VERDICT: PASS or VERDICT: FAIL

Then briefly explain (2-3 paragraphs max):
- Is the correct answer truly correct?
- Are any distractors arguably correct?
- Any logical flaws or inconsistencies?"""

    response = call_gemini(prompt)
    
    if not response["success"]:
        return {
            "passed": None,
            "error": response["error"],
            "gemini_verdict": None,
            "gemini_reasoning": None,
            "adjudication_mode": adjudication_mode
        }
    
    content = response["content"]
    
    # Parse verdict - be more flexible in matching
    verdict = None
    
    # Normalize whitespace for matching
    content_normalized = content.strip()
    
    # Try multiple patterns
    patterns = [
        r'VERDICT:\s*(PASS|FAIL)',           # Standard
        r'\*\*VERDICT:\s*(PASS|FAIL)\*\*',   # Markdown bold
        r'VERDICT\s+(PASS|FAIL)',            # No colon
        r'\n\s*VERDICT:\s*(PASS|FAIL)',      # After newline
        r'VERDICT:\s*(PASS|FAIL)\s*$',       # At end
    ]
    
    for pattern in patterns:
        verdict_match = re.search(pattern, content_normalized, re.IGNORECASE | re.MULTILINE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
            break
    
    if not verdict:
        # Fallback: look for clear indicators in the text
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in ['the question is sound', 'is correct', 'answer is correct', 'scientifically sound', 'suitable for training']):
            if not any(phrase in content_lower for phrase in ['not correct', 'incorrect', 'flawed', 'problematic', 'significant issue', 'ambiguous']):
                verdict = "PASS"
        elif any(phrase in content_lower for phrase in ['incorrect', 'flawed', 'the answer is wrong', 'problematic', 'ambiguous answer', 'not suitable']):
            verdict = "FAIL"
    
    passed = verdict == "PASS"
    
    return {
        "passed": passed,
        "gemini_verdict": verdict,
        "gemini_reasoning": content,
        "adjudication_mode": adjudication_mode,
        "error": None
    }


# === UI FUNCTIONS ===

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def select_domains(domains: list[str]) -> list[str]:
    """Interactive domain selection for expert review."""
    clear_screen()
    print("=" * 70)
    print("  SELECT DOMAINS FOR DIRECT EXPERT REVIEW")
    print("  (These will skip AI screening)")
    print("=" * 70)
    print()
    
    for i, domain in enumerate(domains, 1):
        print(f"  [{i}] {domain}")
    
    print()
    print("Enter numbers separated by commas (e.g., '1,3,5'), or 'none' to skip:")
    print("Enter 'all' to review all questions directly (no AI screening)")
    
    selection = input("\n> ").strip().lower()
    
    if selection == 'none' or selection == '':
        return []
    
    if selection == 'all':
        return domains.copy()
    
    selected = []
    try:
        indices = [int(x.strip()) for x in selection.split(',')]
        for idx in indices:
            if 1 <= idx <= len(domains):
                selected.append(domains[idx - 1])
    except ValueError:
        print("Invalid input. No domains selected for direct review.")
        input("Press Enter to continue...")
        return []
    
    return selected


def display_question(q: dict, num: int, total: int, flags: dict = None):
    """Display a question for review."""
    clear_screen()
    
    print("=" * 70)
    print(f"  QUESTION {num}/{total}  |  Domain: {q['_domain'][:35]}")
    if q.get('_core_concept'):
        print(f"  Core Concept: {q['_core_concept'][:55]}")
    print("=" * 70)
    
    # Show flags if present
    if flags:
        print()
        if flags.get("source") == "expert_domain":
            print("  📋 DIRECT REVIEW (your expert domain)")
        elif flags.get("kimi_disagreed"):
            kimi_ans = flags.get('kimi_answer', '?')
            correct = q.get('correct_letter', '?')
            print(f"  ⚠️  KIMI DISAGREED: picked {kimi_ans} instead of {correct}")
        elif flags.get("kimi_answer") is None and flags.get("source") == "ai_flagged":
            print(f"  ⚠️  KIMI: Could not parse answer from response")
        if flags.get("gemini_failed"):
            print(f"  ⚠️  GEMINI FLAGGED: see reasoning below")
        if flags.get("verification_tag"):
            print(f"  🏷️  Would be tagged: {flags['verification_tag']}")
    
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
    print("ORIGINAL REASONING:")
    thinking = q.get('thinking', '')[:500]
    print(f"  {thinking}{'...' if len(q.get('thinking', '')) > 500 else ''}")
    
    # Show AI reasoning if flagged
    if flags and flags.get("gemini_reasoning"):
        print("-" * 70)
        print("GEMINI'S EVALUATION (truncated, press 'a' for full):")
        gemini_text = flags["gemini_reasoning"][:600]
        print(f"  {gemini_text}{'...' if len(flags.get('gemini_reasoning', '')) > 600 else ''}")
    
    if flags and flags.get("kimi_reasoning"):
        print("-" * 70)
        label = "KIMI'S ALTERNATIVE REASONING" if flags.get("kimi_disagreed") else "KIMI'S REASONING"
        print(f"{label} (truncated, press 'a' for full):")
        kimi_text = flags["kimi_reasoning"][:400]
        print(f"  {kimi_text}{'...' if len(flags.get('kimi_reasoning', '')) > 400 else ''}")
    
    print("=" * 70)


def display_full(q: dict, flags: dict = None):
    """Display full question without truncation."""
    clear_screen()
    
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
    print(f"ORIGINAL REASONING:\n{q.get('thinking', '')}")
    
    print("=" * 70)
    input("\nPress Enter to go back...")


def display_ai_reasoning(q: dict, flags: dict = None):
    """Display full AI screening reasoning."""
    clear_screen()
    
    print("=" * 70)
    print("  FULL AI SCREENING OUTPUT")
    print("=" * 70)
    
    if not flags:
        print("\nNo AI screening data for this question.")
        input("\nPress Enter to go back...")
        return
    
    # Kimi section
    print("\n" + "=" * 70)
    print("KIMI SELF-CHECK:")
    print("=" * 70)
    
    kimi_answer = flags.get('kimi_answer')
    correct = q.get('correct_letter', '?')
    
    if kimi_answer:
        if flags.get('kimi_disagreed'):
            print(f"\n❌ Kimi picked: {kimi_answer} (correct was: {correct})")
        else:
            print(f"\n✓ Kimi agreed: {kimi_answer}")
    else:
        print(f"\n⚠️ Could not parse Kimi's answer")
    
    kimi_reasoning = flags.get('kimi_reasoning', '')
    if kimi_reasoning:
        print(f"\nKIMI'S FULL REASONING:\n{'-' * 40}")
        print(kimi_reasoning)
    else:
        print("\n(No reasoning captured)")
    
    # Gemini section
    print("\n" + "=" * 70)
    print("GEMINI EVALUATION:")
    print("=" * 70)
    
    gemini_verdict = flags.get('gemini_verdict')
    if gemini_verdict:
        symbol = "✓" if gemini_verdict == "PASS" else "❌"
        print(f"\n{symbol} Verdict: {gemini_verdict}")
    else:
        print(f"\n⚠️ Could not parse Gemini's verdict")
    
    if flags.get('adjudication_mode'):
        print("(Adjudication mode: Gemini was told about Kimi's disagreement)")
    
    gemini_reasoning = flags.get('gemini_reasoning', '')
    if gemini_reasoning:
        print(f"\nGEMINI'S FULL RESPONSE:\n{'-' * 40}")
        print(gemini_reasoning)
    else:
        print("\n(No reasoning captured)")
    
    print("\n" + "=" * 70)
    input("\nPress Enter to go back...")


def run_ai_screening(questions: list[dict], expert_domains: list[str]) -> tuple[list, list, list]:
    """
    Run AI screening on questions not in expert domains.
    Returns (expert_queue, flagged_queue, auto_verified).
    """
    expert_queue = []
    to_screen = []
    
    # Separate expert domain questions
    for q in questions:
        if q["_domain"] in expert_domains:
            expert_queue.append({
                "question": q,
                "flags": {
                    "source": "expert_domain",
                    "verification_tag": "expert-verified"
                }
            })
        else:
            to_screen.append(q)
    
    if not to_screen:
        return expert_queue, [], []
    
    print(f"\nRunning AI screening on {len(to_screen)} questions...")
    print("-" * 50)
    
    flagged_queue = []
    auto_verified = []
    
    for i, q in enumerate(to_screen):
        concept = q.get('_core_concept', q['_domain'])[:40]
        print(f"  [{i+1}/{len(to_screen)}] {concept}...")
        
        # Kimi screen
        print("    Kimi check...", end=" ", flush=True)
        kimi_result = kimi_screen(q)
        
        if kimi_result.get("error"):
            print(f"ERROR: {kimi_result['error'][:50]}")
            flagged_queue.append({
                "question": q,
                "flags": {
                    "source": "kimi_error",
                    "error": kimi_result["error"],
                    "verification_tag": "human-verified-flagged"
                }
            })
            time.sleep(1)
            continue
        
        kimi_agreed = kimi_result["passed"]
        print("✓ Agreed" if kimi_agreed else f"✗ Picked {kimi_result.get('kimi_answer_original', '?')}")
        
        time.sleep(5)  # Rate limiting - 5 seconds for Gemini free tier (15 req/min)
        
        # Gemini screen
        print("    Gemini check...", end=" ", flush=True)
        gemini_result = gemini_screen(q, kimi_result if not kimi_agreed else None)
        
        if gemini_result.get("error"):
            print(f"ERROR: {gemini_result['error'][:50]}")
            flagged_queue.append({
                "question": q,
                "flags": {
                    "source": "gemini_error",
                    "error": gemini_result["error"],
                    "kimi_disagreed": not kimi_agreed,
                    "kimi_answer": kimi_result.get("kimi_answer_original"),
                    "kimi_reasoning": kimi_result.get("kimi_reasoning"),
                    "verification_tag": "human-verified-flagged"
                }
            })
            time.sleep(1)
            continue
        
        gemini_passed = gemini_result["passed"]
        gemini_verdict = gemini_result.get("gemini_verdict")
        print(f"{'✓ Pass' if gemini_passed else '✗ Fail'} (verdict: {gemini_verdict})")
        
        if gemini_passed and kimi_agreed:
            # Auto-verify
            auto_verified.append({
                "question": q,
                "flags": {
                    "source": "model_verified",
                    "verification_tag": "model-verified",
                    "kimi_agreed": True,
                    "gemini_passed": True
                }
            })
        else:
            # Flagged for human review
            flagged_queue.append({
                "question": q,
                "flags": {
                    "source": "ai_flagged",
                    "kimi_disagreed": not kimi_agreed,
                    "kimi_answer": kimi_result.get("kimi_answer_original") if not kimi_agreed else None,
                    "kimi_reasoning": kimi_result.get("kimi_reasoning") if not kimi_agreed else None,
                    "gemini_failed": not gemini_passed,
                    "gemini_verdict": gemini_result.get("gemini_verdict"),
                    "gemini_reasoning": gemini_result.get("gemini_reasoning"),
                    "adjudication_mode": gemini_result.get("adjudication_mode"),
                    "verification_tag": "human-verified-flagged"
                }
            })
        
        time.sleep(5)  # Rate limiting - 5 seconds between questions
    
    print("-" * 50)
    print(f"Screening complete:")
    print(f"  Expert review queue: {len(expert_queue)}")
    print(f"  Auto-verified: {len(auto_verified)}")
    print(f"  Flagged for review: {len(flagged_queue)}")
    input("\nPress Enter to begin review...")
    
    return expert_queue, flagged_queue, auto_verified


def review_session(filepath: str):
    """Run a tiered review session."""
    
    questions = load_batch(filepath)
    
    if not questions:
        print("No valid questions found in batch.")
        return
    
    print(f"Loaded {len(questions)} questions from {filepath}")
    
    # Get domains and let user select
    domains = get_domains(questions)
    expert_domains = select_domains(domains)
    
    print(f"\nDirect expert review: {len(expert_domains)} domains")
    for d in expert_domains:
        print(f"  - {d}")
    
    # Run AI screening
    expert_queue, flagged_queue, auto_verified = run_ai_screening(questions, expert_domains)
    
    # Combine review queues
    review_queue = expert_queue + flagged_queue
    
    if not review_queue:
        print("\nNo questions require human review!")
        print(f"Auto-verified: {len(auto_verified)} questions")
        
        if auto_verified:
            save = input("Save auto-verified questions? (y/n): ").strip().lower()
            if save == 'y':
                save_results([], auto_verified, [], filepath)
        return
    
    # Track review status
    reviews = {}
    for item in review_queue:
        idx = item["question"]["_index"]
        reviews[idx] = {"status": "pending", "notes": ""}
    
    current = 0
    
    while True:
        item = review_queue[current]
        q = item["question"]
        flags = item["flags"]
        
        display_question(q, current + 1, len(review_queue), flags)
        
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
        print("  [f] Full view   [a] AI reasoning   [j] Next   [k] Previous   [g] Go to #")
        print("  [s] Summary   [w] Save & quit   [q] Quit without saving")
        
        cmd = input("\n> ").strip().lower()
        
        if cmd == 'v':
            reviews[q['_index']]['status'] = 'verified'
            reviews[q['_index']]['verification_tag'] = flags.get('verification_tag', 'expert-verified')
            print("Marked as VERIFIED")
            if current < len(review_queue) - 1:
                current += 1
        
        elif cmd == 'r':
            reason = input("Rejection reason (optional): ").strip()
            reviews[q['_index']]['status'] = 'rejected'
            reviews[q['_index']]['notes'] = reason
            print("Marked as REJECTED")
            if current < len(review_queue) - 1:
                current += 1
        
        elif cmd == 'e':
            note = input("What needs editing? ").strip()
            reviews[q['_index']]['status'] = 'edit'
            reviews[q['_index']]['notes'] = note
            print("Marked as NEEDS EDIT")
            if current < len(review_queue) - 1:
                current += 1
        
        elif cmd == 'n':
            note = input("Note: ").strip()
            reviews[q['_index']]['notes'] = note
            print("Note added")
        
        elif cmd == 'f':
            display_full(q, flags)
        
        elif cmd == 'a':
            display_ai_reasoning(q, flags)
        
        elif cmd == 'j':
            if current < len(review_queue) - 1:
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
                if 1 <= num <= len(review_queue):
                    current = num - 1
                else:
                    print(f"Invalid. Enter 1-{len(review_queue)}")
                    input("Press Enter...")
            except ValueError:
                print("Invalid number")
                input("Press Enter...")
        
        elif cmd == 's':
            show_summary(review_queue, reviews, auto_verified)
        
        elif cmd == 'w':
            save_results(review_queue, auto_verified, reviews, filepath)
            break
        
        elif cmd == 'q':
            confirm = input("Quit without saving? (y/n): ")
            if confirm.lower() == 'y':
                break


def show_summary(review_queue: list, reviews: dict, auto_verified: list):
    """Show review summary."""
    clear_screen()
    print("=" * 70)
    print("  REVIEW SUMMARY")
    print("=" * 70)
    
    counts = {"pending": 0, "verified": 0, "rejected": 0, "edit": 0}
    for item in review_queue:
        idx = item["question"]["_index"]
        counts[reviews[idx]['status']] += 1
    
    print(f"\n  Human Review Queue ({len(review_queue)} questions):")
    print(f"    ✅ Verified:    {counts['verified']}")
    print(f"    ❌ Rejected:    {counts['rejected']}")
    print(f"    ✏️  Needs edit:  {counts['edit']}")
    print(f"    ⏳ Pending:     {counts['pending']}")
    
    print(f"\n  Auto-verified (model-verified): {len(auto_verified)}")
    
    total_verified = counts['verified'] + len(auto_verified)
    total_questions = len(review_queue) + len(auto_verified)
    print(f"\n  Total verified: {total_verified}/{total_questions}")
    
    # Show rejected/edit items
    print("\n" + "-" * 70)
    print("Rejected/Needs Edit:")
    for i, item in enumerate(review_queue):
        idx = item["question"]["_index"]
        rev = reviews[idx]
        if rev['status'] in ['rejected', 'edit']:
            concept = item["question"].get("_core_concept", "")[:30]
            print(f"  #{i+1} [{rev['status']}] {concept}: {rev['notes'][:40]}")
    
    print("=" * 70)
    input("\nPress Enter to continue...")


def save_results(review_queue: list, auto_verified: list, reviews: dict, original_path: str):
    """Save review results."""
    
    os.makedirs("data/verified", exist_ok=True)
    os.makedirs("data/rejected", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    verified = []
    rejected = []
    
    # Process human-reviewed questions
    for item in review_queue:
        q = item["question"]
        idx = q["_index"]
        rev = reviews.get(idx, {"status": "pending", "notes": ""})
        flags = item["flags"]
        
        clean_q = {k: v for k, v in q.items() if not k.startswith('_')}
        clean_q['domain'] = q.get('_domain', '')
        clean_q['core_concept'] = q.get('_core_concept', '')
        clean_q['review_notes'] = rev.get('notes', '')
        clean_q['verification_tag'] = rev.get('verification_tag', flags.get('verification_tag', ''))
        
        if rev['status'] == 'verified':
            verified.append(clean_q)
        elif rev['status'] in ['rejected', 'edit']:
            clean_q['rejection_reason'] = rev['notes']
            clean_q['review_status'] = rev['status']
            clean_q['ai_flags'] = {
                k: v for k, v in flags.items() 
                if k in ['kimi_disagreed', 'kimi_answer', 'gemini_failed', 'gemini_reasoning']
            }
            rejected.append(clean_q)
    
    # Process auto-verified questions
    for item in auto_verified:
        q = item["question"]
        flags = item["flags"]
        
        clean_q = {k: v for k, v in q.items() if not k.startswith('_')}
        clean_q['domain'] = q.get('_domain', '')
        clean_q['core_concept'] = q.get('_core_concept', '')
        clean_q['verification_tag'] = 'model-verified'
        clean_q['review_notes'] = ''
        verified.append(clean_q)
    
    # Save verified
    if verified:
        outpath = f"data/verified/reviewed_{timestamp}.json"
        with open(outpath, 'w') as f:
            json.dump(verified, f, indent=2)
        print(f"Saved {len(verified)} verified questions to {outpath}")
        
        # Count by tag
        expert_count = sum(1 for v in verified if v.get('verification_tag') == 'expert-verified')
        model_count = sum(1 for v in verified if v.get('verification_tag') == 'model-verified')
        flagged_count = sum(1 for v in verified if v.get('verification_tag') == 'human-verified-flagged')
        print(f"  - expert-verified: {expert_count}")
        print(f"  - model-verified: {model_count}")
        print(f"  - human-verified-flagged: {flagged_count}")
    
    # Save rejected
    if rejected:
        outpath = f"data/rejected/rejected_{timestamp}.json"
        with open(outpath, 'w') as f:
            json.dump(rejected, f, indent=2)
        print(f"Saved {len(rejected)} rejected questions to {outpath}")


def main():
    if len(sys.argv) < 2:
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
        
        print("Usage: python review_tiered.py <batch_file.json>")
        return
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    review_session(filepath)


if __name__ == "__main__":
    main()