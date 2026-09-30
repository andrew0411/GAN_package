"""TensorBoard(+선택적 W&B) 로거. TensorBoard는 생성 시점에, wandb는 `use_wandb=True`일 때만 import한다."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image

from gan_common.viz import denorm

_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')  # Windows 파일명에 쓸 수 없는 문자


class Logger:
    """scalar·이미지 로깅. 산출물: `run_dir/tb/`(TensorBoard), `run_dir/samples/*.png`, W&B run(옵션).

    `with Logger(...) as logger:`로 쓰면 예외가 나도 `close()`가 호출된다.
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        use_wandb: bool = False,
        project: str = "GAN_package",
        run_name: str | None = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.run_dir = Path(run_dir)
        self.sample_dir = self.run_dir / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self._closed = False

        self._wandb = None
        self._run = None
        if use_wandb:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=run_name or self.run_dir.name,
                config=config,
                dir=str(self.run_dir),
            )

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """`{"loss/D": 0.7, ...}` 형태. 값은 float로 바꿔 기록한다 (Tensor면 `.item()`)."""
        values = {k: float(v.item() if isinstance(v, Tensor) else v) for k, v in scalars.items()}
        for k, v in values.items():
            self.writer.add_scalar(k, v, step)
        if self._run is not None:
            self._run.log(values, step=step)

    def log_images(self, tag: str, images: Tensor, step: int, nrow: int = 8) -> None:
        """(N, C, H, W) [-1, 1] 이미지를 grid로 만들어 TensorBoard와 `samples/{tag}_{step:07d}.png`에 남긴다.

        파일명에서는 tag의 `\\ / : * ? " < > |`를 `_`로 바꾼다 (TensorBoard tag는 원래 이름 그대로).
        """
        grid = make_grid(denorm(images.detach().float().cpu()), nrow=nrow)  # (3, H', W') in [0, 1]
        self.writer.add_image(tag, grid, step)
        safe_tag = _INVALID_FILENAME_CHARS.sub("_", tag)
        save_image(grid, str(self.sample_dir / f"{safe_tag}_{step:07d}.png"))
        if self._run is not None:
            array = grid.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()  # (H', W', 3)
            self._run.log({tag: self._wandb.Image(array)}, step=step)

    def close(self) -> None:
        """TensorBoard를 flush 후 닫고 W&B run을 끝낸다. 여러 번 호출해도 안전하다."""
        if self._closed:
            return
        self._closed = True
        self.writer.flush()
        self.writer.close()
        if self._run is not None:
            self._run.finish()
            self._run = None

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
