import argparse
import pathlib

import numpy as np
import safetensors
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="input model path", required=True)
    parser.add_argument("--output_path", type=str, help="embed output path", required=False)

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    output_path = pathlib.Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    bin_file = input_path / "pytorch_model.bin"
    if bin_file.exists():
        torch_bin = torch.load(bin_file.as_posix(), map_location="cpu", mmap=True)
        embeds_pt = torch_bin["model.embed_tokens.weight"]
        embeds_np = embeds_pt.to(torch.float32).numpy
        np.save(output_path / f"model.embed_tokens.weight.npy", embeds_np)
        print(f"find model.embed_tokens.weight in {bin_file.as_posix()}")
        print(f"save model.embed_tokens.weight.npy in {output_path}")
    else:
        sf_files = input_path.glob("*.safetensors")
        for file in sf_files:
            find = False
            with safetensors.safe_open(file, "pt", device="cpu") as f:
                for k in f.keys():
                    if k == "model.embed_tokens.weight":
                        #
                        print(f"find model.embed_tokens.weight in {file.as_posix()}")
                        res = f.get_tensor(k).to(torch.float32).numpy()
                        np.save(output_path / f"model.embed_tokens.weight.npy", res)
                        break
            if find:
                break
        print(f"save model.embed_tokens.weight.npy in {output_path}")
