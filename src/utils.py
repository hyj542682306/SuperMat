from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
from PIL import Image
import torch


def resolve_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    images = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in valid_ext]
    if not images:
        raise ValueError(f"No images found in input directory: {input_path}")
    return images


def parse_image_index(image_path: Path) -> str:
    stem = image_path.stem
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def collect_multi_view_images(case_input_dir: Path) -> List[Path]:
    candidates = []
    for ext in [".webp", ".png", ".jpg", ".jpeg"]:
        candidates.extend(case_input_dir.glob(f"color_*{ext}"))
    if not candidates:
        raise ValueError(f"No color_*.{{webp,png,jpg,jpeg}} found in: {case_input_dir}")

    def sort_key(path: Path) -> Tuple[int, str]:
        idx = parse_image_index(path)
        if idx.isdigit():
            return int(idx), path.name
        return 10**9, path.name

    return sorted(candidates, key=sort_key)


def load_unet_weights(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(checkpoint_path, map_location="cpu")


def load_rgba_image_as_rgb_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGBA").resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]
    rgb = rgb * alpha + 0.5 * (1.0 - alpha)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def load_rgb_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor


def load_mask_tensor(mask_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    mask_np = np.asarray(mask).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    return mask_tensor


def to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().float().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image.permute(0, 2, 3, 1).squeeze(0)
    elif image.ndim == 3:
        image = image.permute(1, 2, 0)
    else:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape: {tuple(image.shape)}")
    image = image.numpy()
    return (image * 255.0).round().astype(np.uint8)


def orm_to_roughness_metallic(orm_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    orm = orm_tensor.detach().float().cpu().clamp(0.0, 1.0)
    if orm.ndim == 4:
        orm = orm.permute(0, 2, 3, 1).squeeze(0)
    elif orm.ndim == 3:
        orm = orm.permute(1, 2, 0)
    else:
        raise ValueError(f"Expected ORM tensor with 3 or 4 dims, got shape: {tuple(orm.shape)}")
    roughness = orm[:, :, 1:2].repeat(1, 1, 3)
    metallic = orm[:, :, 2:3].repeat(1, 1, 3)
    roughness_np = (roughness.numpy() * 255.0).round().astype(np.uint8)
    metallic_np = (metallic.numpy() * 255.0).round().astype(np.uint8)
    return roughness_np, metallic_np


def load_camera_embed_map(meta_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)
    locations = meta.get("locations", [])

    camera_map = {}
    for loc in locations:
        index = str(loc["index"])
        c2w = torch.as_tensor(loc["transform_matrix"], dtype=torch.float32, device=device)
        camera_map[index] = c2w.reshape(-1)
    return camera_map