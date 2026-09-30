"""이미지 시각화: [-1, 1] 텐서 → [0, 1], grid PNG 저장, GIF 생성."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torchvision.utils import save_image


def denorm(x: Tensor) -> Tensor:
    """[-1, 1] → [0, 1]로 옮기고 범위 밖 값은 clamp한다."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_grid(images: Tensor, path: str | Path, nrow: int = 8) -> None:
    """(N, C, H, W) [-1, 1] 이미지를 grid PNG로 저장한다. 상위 폴더가 없으면 만든다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm(images.detach().float().cpu()), str(path), nrow=nrow)


def make_gif(image_paths: list, out_path: str | Path, duration_ms: int = 200) -> None:
    """이미지 파일들을 순서대로 이어 무한 반복 GIF로 저장한다 (PIL `save_all`)."""
    if not image_paths:
        raise ValueError("make_gif: image_paths가 비어 있습니다.")
    frames = []
    for p in image_paths:
        with Image.open(p) as img:
            frames.append(img.convert("RGB"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
