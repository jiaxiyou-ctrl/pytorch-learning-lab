"""Text generation using GPT-2 via HuggingFace pipeline."""

from transformers import pipeline


def run_text_generation():
    """Generate continuations for a list of prompts using GPT-2."""
    MAX_LENGTH = 60
    NUM_RETURN_SEQUENCES = 1

    print("Loading GPT-2...")
    text_generator = pipeline("text-generation", model="gpt2")

    prompts = [
        "Artificial intelligence will",
        "The future of deep learning is",
        "PyTorch is a powerful framework because",
    ]

    print("=" * 60)
    print("  GPT-2 Text Generation")
    print("=" * 60)

    for prompt in prompts:
        results = text_generator(
            prompt,
            max_length=MAX_LENGTH,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            truncation=True,
        )
        print(f'\nPrompt: "{prompt}"')
        for i, result in enumerate(results, start=1):
            generated = result["generated_text"]
            print(f'Output {i}: "{generated}"')


if __name__ == "__main__":
    run_text_generation()
