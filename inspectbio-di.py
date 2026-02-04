from datasets import load_dataset

# Load GPQA Diamond (need to accept terms on HuggingFace first)
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

# Filter to biology only
biology = gpqa["train"].filter(lambda x: "Biology" in x["Subdomain"])

# See what subdomains exist
print(set(gpqa["train"]["Subdomain"]))

# Check how many biology questions
print(f"Biology questions: {len(biology)}")

# Look at one
print(biology[0])