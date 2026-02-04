from datasets import load_dataset
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
print(len(gpqa["train"]))