"""
This script generates text using a pre-trained GPT model.
Functions:
    generate_text(model_dir: str, input_text: str, max_new_tokens: int) -> str:
        Generates text based on the input text using a pre-trained GPT model.
        Args:
            model_dir (str): Path to the saved model checkpoint.
            input_text (str): Input text to generate text from.
            max_new_tokens (int): Maximum number of new tokens to generate.
        Returns:
            str: The generated text.
    main():
        Parses command-line arguments and invokes the text generation function.
Command-line Arguments:
    --model_path (str): Path to the saved model checkpoint. Default is "model/gpt_model.pt".
    --input_text (str): Input text to generate text from. Default is "Every effort moves".
    --max_new_tokens (int): Maximum number of new tokens to generate. Default is 50.
Usage:
    Run the script from the command line with optional arguments:
        python generate.py --model_path <path_to_model> --input_text <input_text> --max_new_tokens <number_of_tokens>
"""

import sys
import os
import torch
import tiktoken as ttk
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from model.configuration import GPTConfig
from model.modeling import GPTModel


def generate_text(model_dir, input_text, max_new_tokens): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint = torch.load(model_dir, map_location=device)

    model = GPTModel(GPTConfig)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print("Model loaded.")


    tokenizer = ttk.get_encoding(GPTConfig.model_type)
    print("Tokenizer loaded.")
    print(f"Input text: {input_text}")      

    encoded = tokenizer.encode(input_text)
    context = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=max_new_tokens)

    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text

def main():
    parser = argparse.ArgumentParser(description="generate text using GPT Moddel.")
    parser.add_argument("--model_path", type=str, default="gpt_model.pt", help="Path to the saved model checkpoint.")
    parser.add_argument("--input_text", type=str, default="Every effort moves ", help="Input text to generate text from.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")

    args = parser.parse_args()

    generated_text = generate_text(args.model_path, args.input_text, args.max_new_tokens)
    print("Generated text:")
    print(generated_text.replace("\n", " "))
    print("Text generated.")


if __name__ == "__main__":
    main()