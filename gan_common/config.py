"""argparse 공통 인자. 폴더별 train.py는 `base_parser(...)`에 고유 인자를 추가해 쓴다."""

from __future__ import annotations

import argparse
from typing import Any

_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


def str2bool(v: str) -> bool:
    """`--flag true/false` 형식의 argparse `type`. bool이 들어오면 그대로 돌려준다."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise argparse.ArgumentTypeError(f"boolean 값이 필요합니다: {v!r}")


def positive_int(v: str) -> int:
    """1 이상의 정수만 받는 argparse `type`. 주기 인자(`step % log_every`)의 ZeroDivisionError를 막는다."""
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"정수가 필요합니다: {v!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"1 이상의 정수가 필요합니다: {v!r}")
    return n


def base_parser(description: str, **defaults: Any) -> argparse.ArgumentParser:
    """공통 학습 인자를 가진 parser. `**defaults`로 폴더별 기본값을 덮어쓴다 (`set_defaults`).

    주의: `**defaults`의 key에 오타가 있어도 에러 없이 namespace에 새 key로 추가된다.
    또 `set_defaults`는 이 함수 안에서 호출되므로, 반환 후 `add_argument`로 추가한 인자에는
    적용되지 않는다 (그런 인자의 기본값은 `add_argument(default=...)`로 준다).
    """
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--data_root", type=str, default=None, help="데이터 루트. 없으면 DATA_ROOT env, 그다음 ~/data")
    g.add_argument("--dataset", type=str, default="mnist", help="mnist | fashion_mnist | cifar10 | celeba | folder | fake")
    g.add_argument("--image_size", type=int, default=64)
    g.add_argument("--channels", type=int, default=1)
    g.add_argument("--download", action="store_true", help="torchvision 데이터셋이 없으면 내려받는다")

    g = p.add_argument_group("optimization")
    g.add_argument("--batch_size", type=int, default=128)
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    g.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")

    g = p.add_argument_group("runtime")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    g.add_argument("--num_workers", type=int, default=4)

    g = p.add_argument_group("logging")
    g.add_argument("--out_dir", type=str, default="runs", help="산출물 루트: <out_dir>/<model>/<run_name>/")
    g.add_argument("--run_name", type=str, default=None, help="없으면 timestamp")
    g.add_argument("--log_every", type=positive_int, default=100, help="scalar 로깅 주기 (step)")
    g.add_argument("--sample_every", type=positive_int, default=500, help="샘플 이미지 저장 주기 (step)")
    g.add_argument("--save_every", type=positive_int, default=1, help="checkpoint 저장 주기 (epoch)")
    g.add_argument("--wandb", action="store_true", help="W&B 로깅 사용")
    g.add_argument("--wandb_project", type=str, default="GAN_package")
    g.add_argument("--resume", type=str, default=None, help="이어서 학습할 checkpoint 경로")

    p.set_defaults(**defaults)
    return p
