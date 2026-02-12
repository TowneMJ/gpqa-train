import getpass
from pathlib import Path

def main():
    print("=== API Key Setup ===\n")

    hf_token = getpass.getpass("Enter your HF_TOKEN: ")
    nvidia_key = getpass.getpass("Enter your NVIDIA_API_KEY: ")
    gemini_key = getpass.getpass("Enter your GEMINI_API_KEY: ")

    env_path = Path(".env")

    with open(env_path, "w") as f:
        f.write(f"HF_TOKEN={hf_token}\n")
        f.write(f"NVIDIA_API_KEY={nvidia_key}\n")
        f.write(f"GEMINI_API_KEY={gemini_key}\n")

    print(f"\n.env file created at {env_path.resolve()}")

if __name__ == "__main__":
    main()