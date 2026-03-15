import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import torch

from diffusers import DDIMScheduler
from src.adapters import UVRefineAdapterWrapper
from src.pipelines.pipeline_uv_refine_stable_diffusion import UVRefineStableDiffusionPipeline
from src.utils import load_unet_weights, load_mask_tensor, load_rgb_tensor, to_uint8_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UV Refine inference script.")
    parser.add_argument("--input-uv", type=str, required=True, help="Path to input UV base-color/ORM map.")
    parser.add_argument("--input-uv-position", type=str, required=True, help="Path to UV position map.")
    parser.add_argument("--input-uv-mask", type=str, required=True, help="Path to UV mask map.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1", help="Base model path or Hugging Face repo id.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--image-size", type=int, default=1024, help="Input resize resolution.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> UVRefineStableDiffusionPipeline:
    pipe = UVRefineStableDiffusionPipeline.from_pretrained(
        args.base_model,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    pipe = UVRefineAdapterWrapper.convert(
        pipe,
        use_camera_embeddings=False,
        camera_embeddings_dim=16,
        replicate_num=1,
    )

    print("Loading UNet weights from checkpoint...")
    unet_weights = load_unet_weights(Path(args.checkpoint))
    incompatible_keys = pipe.unet.load_state_dict(unet_weights, strict=False)
    pipe.unet.eval()
    print(f"Loaded UNet checkpoint from: {args.checkpoint}")
    print(incompatible_keys)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe = pipe.to(args.device)
    return pipe


def run_uv_refine(
    pipe: UVRefineStableDiffusionPipeline,
    input_uv: torch.Tensor,
    input_uv_position: torch.Tensor,
    input_uv_mask: torch.Tensor,
) -> list[torch.Tensor]:
    with torch.no_grad():
        output = pipe(
            prompt="",
            num_inference_steps=1,
            input_uv=input_uv,
            input_uv_position=input_uv_position,
            input_uv_mask=input_uv_mask,
            output_type="pt",
        )
    return output


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    input_uv_path = Path(args.input_uv)
    input_uv_position_path = Path(args.input_uv_position)
    input_uv_mask_path = Path(args.input_uv_mask)

    for path in [input_uv_path, input_uv_position_path, input_uv_mask_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    pipe = build_pipeline(args)
    
    input_uv = load_rgb_tensor(input_uv_path, image_size=args.image_size, device=device)
    input_uv_position = load_rgb_tensor(input_uv_position_path, image_size=args.image_size, device=device)
    input_uv_mask = load_mask_tensor(input_uv_mask_path, image_size=args.image_size, device=device)

    output = run_uv_refine(
        pipe=pipe,
        input_uv=input_uv,
        input_uv_position=input_uv_position,
        input_uv_mask=input_uv_mask,
    )

    image = to_uint8_rgb(output[0])
    output_path = output_dir / "uv_refined.png"
    Image.fromarray(image).save(output_path)


if __name__ == "__main__":
    main()