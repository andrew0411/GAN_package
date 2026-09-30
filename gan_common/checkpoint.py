"""Checkpoint 저장/복원. `torch.load(weights_only=True)`로 읽을 수 있는 값만 넣는다.

넣어도 되는 것: Tensor, state_dict, int/float/str/bool/None, 이들로 된 list/tuple/dict.
넣으면 안 되는 것: nn.Module 객체, argparse.Namespace, Path, numpy scalar/array
(config는 `vars(args)` dict, 수치는 `.item()`·`float()`로 바꿔서 넣는다).
"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

__all__ = ["load_checkpoint", "save_checkpoint", "save_rolling_checkpoint"]

# weights_only 로더가 받아주는 타입. subclass(numpy.float64 ⊂ float, namedtuple ⊂ tuple 등)는
# 거부되므로 isinstance가 아니라 정확한 type으로 비교한다 (Tensor·Parameter만 isinstance).
_SAFE_SCALARS = (int, float, str, bool, type(None))
_SAFE_TORCH = (torch.Size, torch.dtype, torch.device)
_SAFE_MAPPINGS = (dict, OrderedDict)
_SAFE_SEQUENCES = (list, tuple)


def _type_name(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__qualname__}"


def _validate(obj: Any, path: str) -> None:
    """`obj`가 weights_only로 읽을 수 있는 값인지 재귀 검사한다. 아니면 key 경로를 담아 TypeError."""
    t = type(obj)
    if isinstance(obj, torch.Tensor) or t in _SAFE_SCALARS or t in _SAFE_TORCH:
        return
    if t in _SAFE_MAPPINGS:
        for k, v in obj.items():
            key_path = f"{path}[{k!r}]"
            if type(k) not in _SAFE_SCALARS:
                raise TypeError(f"save_checkpoint: '{key_path}'의 key 타입 {_type_name(k)}은 저장할 수 없습니다.")
            _validate(v, key_path)
        return
    if t in _SAFE_SEQUENCES:
        for i, v in enumerate(obj):
            _validate(v, f"{path}[{i}]")
        return
    raise TypeError(
        f"save_checkpoint: '{path}'의 타입 {_type_name(obj)}은 torch.load(weights_only=True)로 읽을 수 없습니다. "
        "Tensor/state_dict/기본 자료형으로 바꾸십시오 (예: vars(args), str(path), .item(), float())."
    )


def save_checkpoint(path: str | Path, **state: Any) -> None:
    """`torch.save(state)`. 저장 전에 weights_only 호환 타입만 있는지 검사하고(아니면 TypeError),
    임시 파일에 쓴 뒤 교체해서 저장 중 중단돼도 기존 파일이 깨지지 않게 한다."""
    for key, value in state.items():
        _validate(value, key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_rolling_checkpoint(ckpt_dir: str | Path, index: int, keep_every: int = 0, **state: Any) -> Path:
    """`<ckpt_dir>/last.pt`를 덮어쓰고, `keep_every > 0`이고 `index % keep_every == 0`이면
    `ckpt_{index:07d}.pt` 사본도 남긴다. `last.pt` 경로를 돌려준다.

    `index`는 호출자가 세는 저장 번호(보통 epoch)다. 사본은 다시 직렬화하지 않고 `last.pt`를 파일 복사한다.
    """
    if keep_every < 0:
        raise ValueError(f"keep_every는 0 이상이어야 합니다: {keep_every}")
    ckpt_dir = Path(ckpt_dir)
    last = ckpt_dir / "last.pt"
    save_checkpoint(last, **state)
    if keep_every > 0 and index % keep_every == 0:
        numbered = ckpt_dir / f"ckpt_{index:07d}.pt"
        tmp = numbered.with_name(numbered.name + ".tmp")
        shutil.copyfile(last, tmp)
        os.replace(tmp, numbered)
    return last


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """`weights_only=True`로 안전하게 읽는다 (임의 코드 실행이 가능한 pickle 객체는 거부)."""
    return torch.load(path, map_location=map_location, weights_only=True)
