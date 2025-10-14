"""
Minimal helper to sanity-check the Hugging Face inference stack once a backend
is wired up. Set `HF_INFERENCE_KEY` in the environment before running.
"""

import os

from huggingface_hub import InferenceClient


def main() -> None:
    client = InferenceClient("HarleyCooper/nanochat", token=os.getenv("HF_INFERENCE_KEY"))
    prompt = "User: Explain why transformers benefit from multi-head attention.\nAssistant:"
    response = client.text_generation(
        prompt,
        max_new_tokens=160,
        temperature=0.7,
        top_p=0.95,
        model="HarleyCooper/nanochat",
    )
    print(response)


if __name__ == "__main__":
    main()
