"""
GPQA-style synthetic data generation using Kimi-k2.5
Skeleton script - fill in prompt details later
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# === CONFIGURATION ===

# Kimi-k2.5 via NVIDIA NIMs (free tier)
# Get your key at: https://build.nvidia.com/moonshotai/kimi-k2.5
CLIENT = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)
MODEL = "moonshotai/kimi-k2.5"

# Alternative: Moonshot API (official)
# CLIENT = OpenAI(
#     api_key=os.getenv("MOONSHOT_API_KEY"),
#     base_url="https://api.moonshot.cn/v1"
# )
# MODEL = "kimi-k2.5"

# Alternative: Together AI
# CLIENT = OpenAI(
#     api_key=os.getenv("TOGETHER_API_KEY"),
#     base_url="https://api.together.xyz/v1"
# )
# MODEL = "moonshotai/Kimi-K2.5"

# Alternative: OpenRouter
# CLIENT = OpenAI(
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1"
# )
# MODEL = "moonshotai/kimi-k2.5"

# Output settings
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === PROMPT TEMPLATE ===
# We'll fill this in later with the full prompt

SYSTEM_PROMPT = """You are a PhD-level molecular biologist creating graduate-level multiple choice questions.

You MUST provide:
1. A challenging question that ends with a clear interrogative sentence (e.g., "Which of the following best explains...", "What is the most likely outcome...", "Which modification would...")
2. Five answer options labeled **A)** through **E)**
3. The correct answer in the format: **Correct Answer: X**
4. An explanation of why the correct answer is right and why others are wrong in the format: **Explanation:**

CRITICAL REQUIREMENTS:
- The question must require MULTI-STEP REASONING, not simple recall. It should involve interpreting experimental results, predicting outcomes, or connecting multiple concepts.
- The question should NOT be answerable by a simple Google search. It should require genuine understanding.
- The question must explicitly ASK something—do not just provide a scenario.
- VARY the position of the correct answer. Do NOT always make B correct. Randomly place the correct answer at A, B, C, D, or E.

Always complete all four parts.
"""

def build_user_prompt(style: str, topic: str, previous_topics: list[str]) -> str:
    """Build the user prompt for a single question generation."""
    
    previous_str = ""
    if previous_topics:
        previous_str = f"\n\nAVOID these already-covered topics:\n- " + "\n- ".join(previous_topics[-20:])
    
    prompt = f"""Generate a challenging graduate-level biology question.

STYLE: {style}
TOPIC AREA: {topic}
{previous_str}

You MUST include ALL of the following:
1. The question text - MUST end with a clear question (e.g., "Which of the following...", "What would be the result...", "Which strategy would best...")
2. Five options labeled **A)** through **E)**
3. **Correct Answer: X** (where X is the letter)
4. **Explanation:** (why correct answer is right, why others are wrong)

IMPORTANT:
- The question must require MULTI-STEP REASONING or interpretation of experimental results—NOT simple factual recall.
- A non-expert should NOT be able to answer it with a quick Google search.
- Do NOT just provide a scenario—you must explicitly ASK a question.
- Place the correct answer randomly at position A, B, C, D, or E (not always B).

Do not stop until you have provided the correct answer and explanation.
"""
    return prompt


# === GENERATION FUNCTIONS ===

def call_kimi(system_prompt: str, user_prompt: str, temperature: float = 1.0) -> dict:
    """
    Call Kimi-k2.5 and return the response.
    Uses thinking mode (temperature=1.0 recommended by Moonshot).
    """
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=0.95,
            max_tokens=8192
        )
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None
        }


def parse_response(raw_content: str) -> dict:
    """
    Parse Kimi's response into structured fields.
    Handles Kimi's natural format: question text, **A)**/**B)**/**C)**/**D)**/**E)** options,
    **Correct Answer:** and **Explanation:**
    """
    import re
    
    result = {
        "thinking": "",
        "question": "",
        "correct_answer": "",
        "incorrect_1": "",
        "incorrect_2": "",
        "incorrect_3": "",
        "incorrect_4": "",
        "raw": raw_content
    }
    
    # Guard against None or non-string content
    if not raw_content or not isinstance(raw_content, str):
        return result
    
    # Try XML tags first (in case we switch to that format later)
    def extract_tag(tag: str, text: str) -> str:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    xml_question = extract_tag("question", raw_content)
    if xml_question:
        # XML format - use original parsing
        result["thinking"] = extract_tag("thinking", raw_content)
        result["question"] = xml_question
        result["correct_answer"] = extract_tag("correct", raw_content)
        result["incorrect_1"] = extract_tag("incorrect_1", raw_content)
        result["incorrect_2"] = extract_tag("incorrect_2", raw_content)
        result["incorrect_3"] = extract_tag("incorrect_3", raw_content)
        result["incorrect_4"] = extract_tag("incorrect_4", raw_content)
        return result
    
    # Otherwise parse Kimi's natural markdown format
    
    # Extract the question (everything before the first option **A)**)
    q_match = re.search(r'^(.*?)(?=\*\*A\))', raw_content, re.DOTALL)
    if q_match:
        result["question"] = q_match.group(1).strip()
    
    # Extract options A, B, C, D, E
    options = {}
    for letter in ['A', 'B', 'C', 'D', 'E']:
        # Match **A)** or **A.** followed by content until next option or section
        pattern = rf'\*\*{letter}[\)\.]?\*\*\s*(.*?)(?=\*\*[B-E][\)\.]?\*\*|\*\*Correct|\*\*\*|$)'
        match = re.search(pattern, raw_content, re.DOTALL)
        if match:
            options[letter] = match.group(1).strip()
    
    # Find correct answer letter
    correct_match = re.search(r'\*\*Correct Answer[:\s]*([A-E])\*\*', raw_content)
    correct_letter = correct_match.group(1) if correct_match else None
    
    # Extract explanation as "thinking"
    expl_match = re.search(r'\*\*Explanation:\*\*(.*?)$', raw_content, re.DOTALL)
    if expl_match:
        result["thinking"] = expl_match.group(1).strip()
    
    # Assign correct and incorrect answers
    if correct_letter and correct_letter in options:
        result["correct_answer"] = options[correct_letter]
        result["correct_letter"] = correct_letter
        
        # Get incorrect options (everything except correct), preserve letters
        incorrect = [(k, options[k]) for k in sorted(options.keys()) if k != correct_letter]
        if len(incorrect) >= 1:
            result["incorrect_1"] = incorrect[0][1]
            result["incorrect_1_letter"] = incorrect[0][0]
        if len(incorrect) >= 2:
            result["incorrect_2"] = incorrect[1][1]
            result["incorrect_2_letter"] = incorrect[1][0]
        if len(incorrect) >= 3:
            result["incorrect_3"] = incorrect[2][1]
            result["incorrect_3_letter"] = incorrect[2][0]
        if len(incorrect) >= 4:
            result["incorrect_4"] = incorrect[3][1]
            result["incorrect_4_letter"] = incorrect[3][0]
    
    return result


def validate_question(parsed: dict) -> tuple[bool, str]:
    """
    Basic validation that all required fields are present.
    Returns (is_valid, reason).
    """
    required = ["question", "correct_answer", "incorrect_1", "incorrect_2", "incorrect_3"]
    
    for field in required:
        if not parsed.get(field):
            return False, f"Missing field: {field}"
    
    # Check minimum lengths
    if len(parsed["question"]) < 50:
        return False, "Question too short"
    
    if len(parsed["correct_answer"]) < 10:
        return False, "Correct answer too short"
    
    return True, "OK"


# === BATCH GENERATION ===

# Question styles to rotate through
STYLES = [
    "Direct address: Start with 'You have...' or 'You are analyzing...'",
    "Third person: Start with 'A researcher...' or 'A scientist...'",
    "Direct question: Start with 'Which of the following...'",
    "Named persona: Start with 'Scientist 1 is studying...'",
    "Subject-first: Start directly with the biological entity or process",
    "Scene-setting: Start with 'In an experiment...' or 'While investigating...'"
]

# Topics to cover - focused on expertise areas
TOPICS = [
    # DNA damage and repair
    "DNA double-strand break repair (NHEJ vs HR)",
    "mismatch repair mechanisms",
    "nucleotide excision repair",
    "base excision repair",
    "translesion synthesis",
    "DNA damage checkpoints",
    "replication stress response",
    
    # Mutagenesis
    "spontaneous mutagenesis mechanisms",
    "chemical mutagens and their effects",
    "UV-induced DNA damage",
    "oxidative DNA damage",
    "frameshift mutations",
    "trinucleotide repeat expansion",
    
    # Genome editing
    "CRISPR-Cas9 mechanisms",
    "CRISPR guide RNA design",
    "base editing",
    "prime editing",
    "off-target effects in genome editing",
    "HDR vs NHEJ in gene editing",
    
    # Molecular genetics
    "transcriptional regulation",
    "promoter and enhancer function",
    "chromatin remodeling",
    "histone modifications",
    "DNA methylation",
    "gene silencing mechanisms",
    
    # DNA replication
    "replication fork dynamics",
    "origin licensing and firing",
    "telomere maintenance",
    "lagging strand synthesis",
    
    # Microbiology/bacterial genetics
    "bacterial DNA replication",
    "bacterial recombination",
    "CRISPR in bacterial immunity",
    "horizontal gene transfer",
    "plasmid biology",
    "bacterial mutagenesis",
    
    # Techniques
    "Southern blotting and hybridization",
    "PCR-based mutagenesis",
    "reporter gene assays",
    "chromatin immunoprecipitation",
]


def generate_batch(
    batch_size: int = 5,
    style: str = None,
    topic: str = None,
    previous_topics: list[str] = None
) -> list[dict]:
    """
    Generate a batch of questions.
    """
    import random
    
    if previous_topics is None:
        previous_topics = []
    
    results = []
    
    for i in range(batch_size):
        # Rotate style if not specified
        current_style = style or STYLES[i % len(STYLES)]
        
        # Pick random topic if not specified
        current_topic = topic or random.choice(TOPICS)
        
        print(f"  Generating {i+1}/{batch_size}: {current_topic[:30]}...")
        
        # Build prompts
        user_prompt = build_user_prompt(current_style, current_topic, previous_topics)
        
        # Call Kimi
        response = call_kimi(SYSTEM_PROMPT, user_prompt)
        
        if not response["success"]:
            print(f"    ERROR: {response['error']}")
            results.append({"success": False, "error": response["error"]})
            continue
        
        # Parse response
        parsed = parse_response(response["content"])
        
        # Validate
        is_valid, reason = validate_question(parsed)
        
        result = {
            "success": is_valid,
            "validation": reason,
            "style": current_style,
            "topic": current_topic,
            "data": parsed,
            "usage": response["usage"]
        }
        
        results.append(result)
        
        # Track topic to avoid repetition
        if is_valid:
            previous_topics.append(current_topic)
        
        # Rate limiting - be nice to the API
        time.sleep(1)
    
    return results


def save_results(results: list[dict], filename: str = None):
    """Save results to a JSON file."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/batch_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved to {filename}")
    return filename


def compile_valid_questions(input_files: list[str], output_file: str):
    """
    Compile all valid questions from multiple batch files into a single training file.
    """
    all_questions = []
    
    for filepath in input_files:
        with open(filepath) as f:
            batch = json.load(f)
        
        for item in batch:
            if item.get("success"):
                q = item["data"]
                all_questions.append({
                    "question": q["question"],
                    "thinking": q["thinking"],
                    "correct_answer": q["correct_answer"],
                    "incorrect_1": q["incorrect_1"],
                    "incorrect_2": q["incorrect_2"],
                    "incorrect_3": q["incorrect_3"],
                    "topic": item["topic"],
                    "style": item["style"]
                })
    
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)
    
    print(f"Compiled {len(all_questions)} valid questions to {output_file}")


# === MAIN ===

if __name__ == "__main__":
    print("=== GPQA Synthetic Data Generator ===\n")
    
    # Test with a small batch
    print("Generating test batch of 20 questions...")
    results = generate_batch(batch_size=20)
    
    # Report results
    valid = sum(1 for r in results if r.get("success"))
    print(f"\nResults: {valid}/{len(results)} valid")
    
    for i, r in enumerate(results):
        status = "✓" if r.get("success") else "✗"
        reason = r.get("validation", r.get("error", "unknown"))
        print(f"  {i+1}. {status} {reason}")
    
    # Save
    outfile = save_results(results)
    
    print(f"\nDone! Check {outfile}")