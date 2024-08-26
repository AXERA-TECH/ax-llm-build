import argparse
import logging
import pathlib
from typing import Optional

import numpy as np
import safetensors
import torch


def setup_logging() -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def save_embed_tokens_weight(
    input_path: pathlib.Path, output_path: pathlib.Path
) -> None:
    """
    Extract and save 'model.embed_tokens.weight' from either a .bin or .safetensors file.

    Args:
        input_path (pathlib.Path): The path to the input model directory.
        output_path (pathlib.Path): The path to save the output .npy file.

    Raises:
        FileNotFoundError: If no valid model file is found in the input path.
    """
    bin_file = input_path / "pytorch_model.bin"
    
    if bin_file.exists():
        logging.info(f"Found model in {bin_file}")
        try:
            torch_bin = torch.load(bin_file.as_posix(), map_location="cpu", mmap=True)
            embeds_pt = torch_bin["model.embed_tokens.weight"]
            embeds_np = embeds_pt.to(torch.float32).numpy()
            np.save(output_path / "model.embed_tokens.weight.npy", embeds_np)
            logging.info(f"Saved 'model.embed_tokens.weight.npy' in {output_path}")
        except Exception as e:
            logging.error(f"Failed to process .bin file: {e}")
    else:
        sf_files = list(input_path.glob("*.safetensors"))
        if not sf_files:
            raise FileNotFoundError("No valid model files found in the input path.")
        
        for file in sf_files:
            try:
                with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k == "model.embed_tokens.weight":
                            logging.info(f"Found 'model.embed_tokens.weight' in {file.as_posix()}")
                            res = f.get_tensor(k).to(torch.float32).numpy()
                            np.save(output_path / "model.embed_tokens.weight.npy", res)
                            logging.info(f"Saved 'model.embed_tokens.weight.npy' in {output_path}")
                            return
            except Exception as e:
                logging.error(f"Failed to process {file.as_posix()}: {e}")

        raise FileNotFoundError("No 'model.embed_tokens.weight' found in any safetensors file.")


def main(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Main function to process the input model files and save the extracted weights.

    Args:
        input_path (str): The path to the input model directory.
        output_path (Optional[str]): The path to save the output .npy file. If not provided, defaults to input path.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path) if output_path else input_path / "output"

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    save_embed_tokens_weight(input_path, output_path)


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description="Extract and save model embeddings from a model file.")
    parser.add_argument("--input_path", type=str, help="Input model directory path", required=True)
    parser.add_argument("--output_path", type=str, help="Output directory path for embeddings", required=False)

    args = parser.parse_args()
    main(args.input_path, args.output_path)
