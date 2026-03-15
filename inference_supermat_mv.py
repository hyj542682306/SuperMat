import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch

from diffusers import DDIMScheduler
from src.adapters import SuperMatAdapterWrapper
from src.pipelines.pipeline_supermat_stable_diffusion import SuperMatStableDiffusionPipeline
from src.models.model_utils import set_supermat_mv_self_attention
from src.utils import parse_image_index, collect_multi_view_images, load_unet_weights, load_rgba_image_as_rgb_tensor, to_uint8_rgb, orm_to_roughness_metallic, load_camera_embed_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SuperMat multi-view inference script.")
    parser.add_argument("--input", type=str, required=True, help="Input case directory that contains multi-view images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/supermat_mv.ckpt', help="Checkpoint file path.")
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1", help="Base model path or Hugging Face repo id.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--image-size", type=int, default=512, help="Input resize resolution.")
    parser.add_argument("--num_views", type=int, default=6, help="Number of views per inference batch.")
    parser.add_argument("--use-camera-embeds", action="store_true", help="Use camera embeddings from meta.json transform_matrix.")
    parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers attention processor for MV self-attention.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--save-orm", action="store_true", help="Also save orm output.")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> SuperMatStableDiffusionPipeline:
    pipe = SuperMatStableDiffusionPipeline.from_pretrained(
        args.base_model,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    pipe = SuperMatAdapterWrapper.convert(
        pipe,
        use_camera_embeddings=args.use_camera_embeds,
        camera_embeddings_dim=16,
        replicate_num=2,
    )

    set_supermat_mv_self_attention(
        pipe.unet,
        num_views=args.num_views,
        use_xformers=not args.disable_xformers,
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


def run_case(
    pipe: SuperMatStableDiffusionPipeline,
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    num_views: int,
    use_camera_embeds: bool,
    seed: Optional[int],
    save_orm: bool,
    device: torch.device,
) -> None:
    image_paths = collect_multi_view_images(input_dir)
    assert len(image_paths) == num_views, f"Number of images ({len(image_paths)}) does not match --num_views ({num_views}) for case: {input_dir}"

    meta_path = input_dir / "meta.json"
    camera_map = load_camera_embed_map(meta_path, device) if use_camera_embeds else {}

    source_images = torch.cat(
        [load_rgba_image_as_rgb_tensor(path, image_size=image_size, device=device) for path in image_paths],
        dim=0,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed)

    camera_embeds = None
    if use_camera_embeds:
        embeds = []
        for image_path in image_paths:
            image_idx = parse_image_index(image_path)
            if image_idx not in camera_map:
                raise KeyError(f"Image index {image_idx} is missing in meta.json for case {input_dir.name}")
            embeds.append(camera_map[image_idx])
        camera_embeds = torch.stack(embeds, dim=0)

    with torch.no_grad():
        outputs = pipe(
            prompt="",
            num_inference_steps=1,
            num_images_per_prompt=num_views,
            source_image=source_images,
            output_type="pt",
            camera_embeds=camera_embeds,
            generator=generator,
        )

    albedo_branch = outputs[0]
    orm_branch = outputs[1]

    output_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, image_path in enumerate(image_paths):
        image_index = parse_image_index(image_path)
        albedo = to_uint8_rgb(albedo_branch[local_idx])
        orm = to_uint8_rgb(orm_branch[local_idx])
        roughness, metallic = orm_to_roughness_metallic(orm_branch[local_idx])

        Image.fromarray(albedo).save(output_dir / f"albedo_{image_index}.png")
        Image.fromarray(roughness).save(output_dir / f"roughness_{image_index}.png")
        Image.fromarray(metallic).save(output_dir / f"metallic_{image_index}.png")
        if save_orm:
            Image.fromarray(orm).save(output_dir / f"orm_{image_index}.png")


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

    device = torch.device(args.device)

    pipe = build_pipeline(args)

    run_case(
        pipe=pipe,
        input_dir=Path(args.input),
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        num_views=args.num_views,
        use_camera_embeds=args.use_camera_embeds,
        seed=args.seed,
        save_orm=args.save_orm,
        device=device,
    )


if __name__ == "__main__":
    main()