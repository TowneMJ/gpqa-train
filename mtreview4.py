"""
Multi-tier review script v3 for GPQA-style questions.
Flow:
1. Human selects confident domains for direct review
2. Remaining questions go through:
   a. Gemini critique (identifies any issues)
   b. Kimi validation (agrees/disagrees with Gemini's assessment)
3. Auto-pass if Gemini finds no issues AND Kimi agrees
4. Otherwise, human review with AI notes
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
GEMINI_MODEL = "gemini-2.0-flash"


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


# === API CALL FUNCTIONS ===

def call_kimi(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> dict:
    """Call Kimi API."""
    try:
        response = KIMI_CLIENT.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": "You are a PhD-level scientific reviewer. Your task is to review and scrutinize potential graduate-level test questions. You will receive a candidate question, the intended correct answer, a list of distractors, and a reasoning explanation. You will also receive an assessment of the question's quality by an AI reviewer, Gemini, for further context. Evaluate the provided materials, searching for issues of scientific innacuracy, logical inconsistency, ambiguity, reliance on pure recall rather than reasoning, and general poor quality. Initially state your verdict as either NO ISSUES or ISSUES FOUND on its own line, then briefly explain your critique in 3-5 sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        message = response.choices[0].message
        content = message.content
        reasoning = None
        
        try:
            msg_dict = message.model_dump() if hasattr(message, 'model_dump') else message.__dict__
            reasoning = msg_dict.get('reasoning_content') or msg_dict.get('reasoning')
        except:
            pass
        
        if not content and reasoning:
            return {
                "success": True, 
                "content": None,
                "reasoning": reasoning,
                "note": "No final answer, only reasoning"
            }
        
        return {"success": True, "content": content, "reasoning": reasoning}
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
                wait_time = (attempt + 1) * 30
                print(f"Rate limited, waiting {wait_time}s...", end=" ", flush=True)
                time.sleep(wait_time)
                continue
            return {"success": False, "error": error_str}


# === SCREENING FUNCTIONS ===

def format_question_for_review(q: dict) -> str:
    """Format a question with all its components for AI review."""
    correct_letter = q.get("correct_letter", "?")
    correct_answer = q.get("correct_answer", "")
    
    options_text = f"{correct_letter}) {correct_answer}\n"
    for i in range(1, 5):
        key = f"incorrect_{i}"
        letter_key = f"incorrect_{i}_letter"
        if q.get(key):
            options_text += f"{q.get(letter_key, '?')}) {q[key]}\n"
    
    return f"""QUESTION:
{q.get('question', '')}

OPTIONS:
{options_text}
STATED CORRECT ANSWER: {correct_letter}

REASONING PROVIDED:
{q.get('thinking', '')}"""


def gemini_critique(q: dict) -> dict:
    """
    Have Gemini critique the question for logical, factual, or structural issues.
    Returns dict with 'has_issues', 'critique', 'error'.
    """
    question_text = format_question_for_review(q)
    
    prompt = f"""You are a PhD-level scientist reviewing a graduate exam question for quality.

{question_text}

---

Carefully evaluate this question for any issues:
1. Are there any factual or scientific inaccuracies?
2. Is the stated correct answer actually correct and unambiguous?
3. Are any of the distractors arguably correct or ambiguous?
4. Are there logical inconsistencies in the question setup or reasoning?
5. Are there any experimental conditions mentioned but not used?

If you find ANY issues, describe each one specifically and clearly.

If the question is sound with no issues, respond with exactly: NO ISSUES FOUND

Be thorough but honest. Do not invent problems that don't exist."""

    response = call_gemini(prompt)
    
    if not response["success"]:
        return {
            "has_issues": None,
            "critique": None,
            "error": response["error"]
        }
    
    content = response["content"]
    
    if not content:
        return {
            "has_issues": None,
            "critique": None,
            "error": "Empty response from Gemini"
        }
    
    # Check if Gemini found no issues
    content_clean = content.strip().upper()
    has_issues = "NO ISSUES FOUND" not in content_clean
    
    return {
        "has_issues": has_issues,
        "critique": content,
        "error": None
    }


def kimi_validate(q: dict, gemini_critique: str, gemini_has_issues: bool) -> dict:
    """
    Have Kimi validate Gemini's assessment.
    Returns dict with 'agrees', 'response', 'error'.
    """
    question_text = format_question_for_review(q)
    
    if gemini_has_issues:
        gemini_summary = f"Gemini found the following issues:\n{gemini_critique}"
    else:
        gemini_summary = "Gemini found NO ISSUES with this question."
    
    prompt = f'''Evaluate this test question and Gemini's assessment.

QUESTION UNDER REVIEW:
{question_text}

GEMINI'S ASSESSMENT:
{gemini_summary}

State NO ISSUES or ISSUES FOUND on the first line, then briefly explain why in 3-5 sentences.'''

    response = call_kimi(prompt, max_tokens=8192, temperature=0.1)
    
    if not response["success"]:
        return {
            "agrees": None,
            "response": None,
            "error": response["error"]
        }
    
    content = response.get("content")
    reasoning = response.get("reasoning")
    
    text_to_parse = content if content else reasoning
    
    if not text_to_parse or not isinstance(text_to_parse, str):
        return {
            "agrees": None,
            "response": f"No parseable response. Content: {repr(content)[:50]}, Reasoning: {repr(reasoning)[:50]}",
            "error": "Could not get Kimi response"
        }
    
    text_clean = text_to_parse.strip().upper()
    
    if "AGREE" in text_clean and "DISAGREE" not in text_clean:
        return {
            "agrees": True,
            "response": content or "(from reasoning)",
            "error": None
        }
    
    if "DISAGREE" in text_clean:
        return {
            "agrees": False,
            "response": content or text_to_parse[:200],
            "error": None
        }
    
    return {
        "agrees": None,
        "response": text_to_parse[:300],
        "error": "Could not determine agreement"
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
        else:
            # Show screening results
            if flags.get("gemini_has_issues"):
                print("  ⚠️  GEMINI FOUND ISSUES")
            else:
                print("  ✓  Gemini: No issues found")
            
            if flags.get("kimi_agrees") is True:
                print("  ✓  Kimi: Agrees with Gemini")
            elif flags.get("kimi_agrees") is False:
                print("  ⚠️  KIMI DISAGREES WITH GEMINI")
            elif flags.get("kimi_error"):
                print(f"  ⚠️  Kimi error: {flags.get('kimi_error', '')[:40]}")
        
        if flags.get("verification_tag"):
            print(f"  🏷️  Tag: {flags['verification_tag']}")
    
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
    
    # Show AI critique if present
    if flags and flags.get("gemini_critique"):
        print("-" * 70)
        print("GEMINI'S CRITIQUE:")
        critique = flags["gemini_critique"][:600]
        print(f"  {critique}{'...' if len(flags.get('gemini_critique', '')) > 600 else ''}")
    
    if flags and flags.get("kimi_response") and flags.get("kimi_agrees") is False:
        print("-" * 70)
        print("KIMI'S DISAGREEMENT:")
        print(f"  {flags['kimi_response'][:300]}")
    
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


def display_ai_analysis(q: dict, flags: dict = None):
    """Display full AI screening analysis."""
    clear_screen()
    
    print("=" * 70)
    print("  FULL AI ANALYSIS")
    print("=" * 70)
    
    if not flags:
        print("\nNo AI screening data for this question.")
        input("\nPress Enter to go back...")
        return
    
    # Gemini section
    print("\n" + "=" * 70)
    print("GEMINI CRITIQUE:")
    print("=" * 70)
    
    if flags.get("gemini_has_issues"):
        print("\n⚠️  Gemini found issues:\n")
    else:
        print("\n✓ Gemini found no issues:\n")
    
    print(flags.get("gemini_critique", "(No critique captured)"))
    
    # Kimi section
    print("\n" + "=" * 70)
    print("KIMI VALIDATION:")
    print("=" * 70)
    
    if flags.get("kimi_agrees") is True:
        print("\n✓ Kimi AGREES with Gemini's assessment")
    elif flags.get("kimi_agrees") is False:
        print("\n⚠️  Kimi DISAGREES with Gemini")
    else:
        print("\n⚠️  Could not determine Kimi's position")
    
    kimi_response = flags.get("kimi_response", "")
    if kimi_response:
        print(f"\nKimi's response:\n{kimi_response}")
    
    if flags.get("kimi_error"):
        print(f"\nKimi error: {flags['kimi_error']}")
    
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
        print(f"  [{i+1}/{len(to_screen)}] {concept}")
        
        # Step 1: Gemini critique
        print("    Gemini critique...", end=" ", flush=True)
        gemini_result = gemini_critique(q)
        
        if gemini_result.get("error"):
            print(f"ERROR: {gemini_result['error'][:40]}")
            flagged_queue.append({
                "question": q,
                "flags": {
                    "source": "gemini_error",
                    "error": gemini_result["error"],
                    "verification_tag": "human-review-needed"
                }
            })
            time.sleep(2)
            continue
        
        gemini_has_issues = gemini_result["has_issues"]
        gemini_critique_text = gemini_result["critique"]
        
        if gemini_has_issues:
            print("⚠️  Issues found")
        else:
            print("✓ No issues")
        
        time.sleep(2)  # Rate limiting
        
        # Step 2: Kimi validation
        print("    Kimi validation...", end=" ", flush=True)
        kimi_result = kimi_validate(q, gemini_critique_text, gemini_has_issues)
        
        if kimi_result.get("error"):
            print(f"ERROR: {kimi_result['error'][:40]}")
            # Still usable, just without Kimi validation
            kimi_agrees = None
        else:
            kimi_agrees = kimi_result["agrees"]
            if kimi_agrees:
                print("✓ Agrees")
            else:
                print("⚠️  Disagrees")
        
        time.sleep(2)  # Rate limiting
        
        # Determine outcome
        # Auto-pass only if: Gemini found no issues AND Kimi agrees
        if not gemini_has_issues and kimi_agrees is True:
            auto_verified.append({
                "question": q,
                "flags": {
                    "source": "model_verified",
                    "verification_tag": "model-verified",
                    "gemini_has_issues": False,
                    "gemini_critique": gemini_critique_text,
                    "kimi_agrees": True,
                    "kimi_response": kimi_result.get("response", "")
                }
            })
        else:
            # Flag for human review
            flagged_queue.append({
                "question": q,
                "flags": {
                    "source": "ai_flagged",
                    "verification_tag": "human-review-needed",
                    "gemini_has_issues": gemini_has_issues,
                    "gemini_critique": gemini_critique_text,
                    "kimi_agrees": kimi_agrees,
                    "kimi_response": kimi_result.get("response", ""),
                    "kimi_error": kimi_result.get("error")
                }
            })
    
    print("-" * 50)
    print(f"Screening complete:")
    print(f"  Expert review queue: {len(expert_queue)}")
    print(f"  Auto-verified: {len(auto_verified)}")
    print(f"  Flagged for review: {len(flagged_queue)}")
    input("\nPress Enter to begin review...")
    
    return expert_queue, flagged_queue, auto_verified

def browse_auto_verified(auto_verified: list):
    """Browse questions that passed both AI models."""
    if not auto_verified:
        print("\nNo auto-verified questions to browse.")
        input("Press Enter to continue...")
        return
    
    current = 0
    
    while True:
        item = auto_verified[current]
        q = item["question"]
        flags = item["flags"]
        
        display_question(q, current + 1, len(auto_verified), flags)
        
        print(f"\nAuto-verified ({current + 1}/{len(auto_verified)})")
        print("\nCommands:")
        print("  [f] Full view   [a] AI analysis   [j] Next   [k] Previous   [g] Go to #")
        print("  [r] Revoke (move to human review)   [b] Back to review queue")
        
        cmd = input("\n> ").strip().lower()
        
        if cmd == 'f':
            display_full(q, flags)
        
        elif cmd == 'a':
            display_ai_analysis(q, flags)
        
        elif cmd == 'j':
            if current < len(auto_verified) - 1:
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
                if 1 <= num <= len(auto_verified):
                    current = num - 1
                else:
                    print(f"Invalid. Enter 1-{len(auto_verified)}")
                    input("Press Enter...")
            except ValueError:
                print("Invalid number")
                input("Press Enter...")
        
        elif cmd == 'r':
            confirm = input("Revoke auto-verification and move to human review? (y/n): ").strip().lower()
            if confirm == 'y':
                revoked = auto_verified.pop(current)
                revoked["flags"]["source"] = "revoked_auto"
                revoked["flags"]["verification_tag"] = "human-review-needed"
                if current >= len(auto_verified) and current > 0:
                    current -= 1
                if not auto_verified:
                    print("No more auto-verified questions.")
                    input("Press Enter...")
                    return revoked
                return revoked
        
        elif cmd == 'b':
            return None
    
    return None

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
        print(f"\nNo questions flagged for human review.")
        print(f"Auto-verified: {len(auto_verified)} questions")
        print("\nCommands:")
        print("  [m] Browse model-verified questions")
        print("  [w] Save & quit   [q] Quit without saving")
        
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'm':
                revoked = browse_auto_verified(auto_verified)
                if revoked:
                    review_queue.append(revoked)
                    idx = revoked["question"]["_index"]
                    reviews[idx] = {"status": "pending", "notes": "revoked from auto-verified"}
                    print(f"Moved to review queue. Breaking to full review mode...")
                    input("Press Enter...")
                    break
                # Reprint menu after browsing
                print(f"\nAuto-verified: {len(auto_verified)} questions")
                print("\nCommands:")
                print("  [m] Browse model-verified questions")
                print("  [w] Save & quit   [q] Quit without saving")
            
            elif cmd == 'w':
                save_results([], auto_verified, {}, filepath)
                return
            
            elif cmd == 'q':
                confirm = input("Quit without saving? (y/n): ").strip().lower()
                if confirm == 'y':
                    return
        
            # If we broke out due to a revoke, fall through to the main review loop
    
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
        print("  [f] Full view   [a] AI analysis   [j] Next   [k] Previous   [g] Go to #")
        print(f"  [m] Browse model-verified ({len(auto_verified)})")
        print("  [s] Summary   [w] Save & quit   [q] Quit without saving")
        
        cmd = input("\n> ").strip().lower()
        
        if cmd == 'v':
            reviews[q['_index']]['status'] = 'verified'
            reviews[q['_index']]['verification_tag'] = flags.get('verification_tag', 'human-verified')
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
            display_ai_analysis(q, flags)
        
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
        
        elif cmd == 'm':
            revoked = browse_auto_verified(auto_verified)
            if revoked:
                # Add revoked question to the review queue
                review_queue.append(revoked)
                idx = revoked["question"]["_index"]
                reviews[idx] = {"status": "pending", "notes": "revoked from auto-verified"}
                print(f"Moved to review queue (now {len(review_queue)} questions)")
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
            clean_q['ai_critique'] = {
                'gemini_critique': flags.get('gemini_critique'),
                'gemini_has_issues': flags.get('gemini_has_issues'),
                'kimi_agrees': flags.get('kimi_agrees'),
                'kimi_response': flags.get('kimi_response')
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
        human_count = sum(1 for v in verified if v.get('verification_tag') == 'human-verified')
        print(f"  - expert-verified: {expert_count}")
        print(f"  - model-verified: {model_count}")
        print(f"  - human-verified: {human_count}")
    
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
        
        print("Usage: python mtreview3.py <batch_file.json>")
        return
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    review_session(filepath)


if __name__ == "__main__":
    main()