import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch

from diffusers import DDIMScheduler
from src.adapters import SuperMatAdapterWrapper
from src.pipelines.pipeline_supermat_stable_diffusion import SuperMatStableDiffusionPipeline
from src.utils import resolve_inputs, load_unet_weights, load_rgba_image_as_rgb_tensor, to_uint8_rgb, orm_to_roughness_metallic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SuperMat inference script.")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/supermat.ckpt', help="Checkpoint file path.")
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1", help="Base Stable Diffusion model path or Hugging Face repo id.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--image-size", type=int, default=512, help="Input resize resolution.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--save-orm", action="store_true", help="Also save orm output map.")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> SuperMatStableDiffusionPipeline:
    pipe = SuperMatStableDiffusionPipeline.from_pretrained(
        args.base_model,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    pipe = SuperMatAdapterWrapper.convert(
        pipe,
        use_camera_embeddings=False,
        camera_embeddings_dim=16
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


def run_one_image(
    pipe: SuperMatStableDiffusionPipeline,
    image_path: Path,
    output_dir: Path,
    image_size: int,
    seed: Optional[int],
    save_orm: bool,
    device: torch.device,
) -> None:
    image_tensor = load_rgba_image_as_rgb_tensor(image_path=image_path, image_size=image_size, device=device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed)

    with torch.no_grad():
        images = pipe(
            prompt="",
            num_inference_steps=1,
            source_image=image_tensor,
            output_type="pt",
            generator=generator,
        )

    albedo = to_uint8_rgb(images[0])
    orm = to_uint8_rgb(images[1])
    roughness, metallic = orm_to_roughness_metallic(images[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(albedo).save(output_dir / "albedo.png")
    Image.fromarray(roughness).save(output_dir / "roughness.png")
    Image.fromarray(metallic).save(output_dir / "metallic.png")
    if save_orm:
        Image.fromarray(orm).save(output_dir / "orm.png")


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_root = Path(args.output_dir)
    device = torch.device(args.device)

    pipe = build_pipeline(args)
    image_paths = resolve_inputs(input_path)
    print(f"Found {len(image_paths)} image(s).")

    for idx, image_path in enumerate(image_paths, start=1):
        sample_name = image_path.stem
        sample_out_dir = output_root / sample_name
        run_one_image(
            pipe=pipe,
            image_path=image_path,
            output_dir=sample_out_dir,
            image_size=args.image_size,
            seed=args.seed,
            save_orm=args.save_orm,
            device=device,
        )


if __name__ == "__main__":
    main()