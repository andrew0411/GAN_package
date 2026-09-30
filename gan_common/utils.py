"""실험 공통 유틸리티: seed 고정, device 선택, 파라미터 수, run 디렉터리, config 저장."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """random, numpy, torch(CPU/CUDA) seed를 고정한다.

    `deterministic=True`면 cuDNN conv 알고리즘을 deterministic으로 고정하고 benchmark를 끈다(속도 손실).
    이것으로 결정적이 되는 것은 cuDNN conv뿐이다. CUDA에서 interpolate·pooling backward 등
    일부 연산은 여전히 nondeterministic일 수 있다. 기본값 False는 benchmark=True로 속도를 우선한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(pref: str = "auto") -> torch.device:
    """`auto`면 CUDA 사용 가능 시 cuda, 아니면 cpu. 그 외 값(`cpu`, `cuda:1` 등)은 그대로 쓴다."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(pref)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={pref!r}를 요청했지만 CUDA를 사용할 수 없습니다. --device cpu 또는 auto를 쓰십시오.")
    return device


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    """파라미터 원소 수. `trainable_only=True`면 requires_grad인 것만 센다."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def make_run_dir(out_dir: str | Path, model_name: str, run_name: str | None = None) -> Path:
    """`<out_dir>/<model_name>/<run_name|timestamp>/`와 하위 `samples/`, `checkpoints/`를 만든다."""
    name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir).expanduser() / model_name / name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dir(args: argparse.Namespace, model_name: str) -> Path:
    """이번 실행의 run 디렉터리를 정한다.

    - `--resume`이 있고 `--run_name`이 없으면: checkpoint가 `<run_dir>/checkpoints/` 안에 있다는 전제로
      `<run_dir>`(= checkpoint 경로의 parent.parent)를 재사용해 로그·샘플을 이어 쓴다.
      `samples/`, `checkpoints/`가 없으면 만든다.
    - 그 외: `make_run_dir(args.out_dir, model_name, args.run_name)`로 새로(또는 지정 이름으로) 만든다.
    """
    resume = getattr(args, "resume", None)
    if resume and getattr(args, "run_name", None) is None:
        ckpt = Path(resume).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"--resume checkpoint가 없습니다: '{ckpt}'")
        if ckpt.parent.name != "checkpoints":
            raise ValueError(
                f"--resume 경로가 <run_dir>/checkpoints/ 아래가 아닙니다: '{ckpt}'. "
                "run 디렉터리를 새로 만들려면 --run_name을 함께 주십시오."
            )
        run_dir = ckpt.parent.parent
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        return run_dir
    return make_run_dir(args.out_dir, model_name, getattr(args, "run_name", None))


def save_config(config: dict[str, Any], run_dir: Path) -> None:
    """`run_dir/config.json`에 저장한다. JSON 직렬화가 안 되는 값(Path 등)은 str로 바꾼다.

    기존 `config.json`(원래 run의 설정)은 절대 덮어쓰지 않는다. 이미 있으면(--resume으로 같은 run에
    이어 쓰는 경우 등) `config_resume_<YYYYmmdd-HHMMSS>.json`에 저장한다. 그 이름도 있으면 `_1`, `_2`…를 붙인다.
    """
    run_dir = Path(run_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    names = ["config.json", f"config_resume_{stamp}.json"]
    names += (f"config_resume_{stamp}_{i}.json" for i in range(1, 100))
    for name in names:
        try:
            f = (run_dir / name).open("x", encoding="utf-8")  # "x": 파일이 있으면 FileExistsError
        except FileExistsError:
            continue
        with f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        return
    raise FileExistsError(f"'{run_dir}'에 config_resume_{stamp}*.json 이름이 모두 사용 중입니다.")
