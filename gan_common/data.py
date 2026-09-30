"""데이터 로딩: DATA_ROOT 추상화, 이미지 transform, 데이터셋/로더, pix2pix·CycleGAN용 paired/unpaired 데이터셋.

모든 이미지 텐서는 [-1, 1] 범위(Normalize mean=0.5, std=0.5)로 나온다. Generator 출력의 Tanh와 맞춘다.
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp")

_TORCHVISION_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}
_MEAN_STD_RGB = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_data_root(override: str | None = None) -> Path:
    """데이터 루트: `override` → `DATA_ROOT` 환경변수 → `~/data` 순서로 정한다."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("DATA_ROOT")
    return Path(env).expanduser() if env else Path.home() / "data"


def _to_rgb(img: Image.Image) -> Image.Image:
    """grayscale 원본(MNIST 등)을 3채널로 쓸 때를 위한 변환. module-level이라 worker pickling이 된다."""
    return img.convert("RGB")


def image_transform(
    image_size: int,
    channels: int,
    *,
    center_crop: bool = False,
    random_flip: bool = False,
) -> T.Compose:
    """PIL 이미지 → (C, H, W) Tensor in [-1, 1].

    - `center_crop=True`: 짧은 변을 `image_size`로 Resize 후 CenterCrop (종횡비 유지, CelebA 등)
    - `center_crop=False`: `(image_size, image_size)`로 바로 Resize (MNIST·CIFAR처럼 정사각 데이터)
    - `channels=1`이면 Grayscale, 3이면 RGB로 맞춘다
    """
    if channels not in (1, 3):
        raise ValueError(f"channels는 1 또는 3이어야 합니다: {channels}")
    tfs: list = [T.Grayscale(num_output_channels=1) if channels == 1 else T.Lambda(_to_rgb)]
    if center_crop:
        tfs += [
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(image_size),
        ]
    else:
        tfs.append(T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True))
    if random_flip:
        tfs.append(T.RandomHorizontalFlip())
    tfs += [T.ToTensor(), T.Normalize((0.5,) * channels, (0.5,) * channels)]
    return T.Compose(tfs)


def build_image_dataset(
    name: str,
    image_size: int,
    channels: int,
    *,
    train: bool = True,
    root: str | None = None,
    download: bool = False,
    path: str | None = None,
    num_fake: int = 256,
    classes: list[int] | None = None,
) -> Dataset:
    """이름으로 이미지 데이터셋을 만든다. 각 샘플은 `(img, label)`.

    - `mnist | fashion_mnist | cifar10`: torchvision 데이터셋 (`<data_root>` 아래)
    - `celeba`: `ImageFolder(<data_root>/celeba)` — 예: `celeba/img_align_celeba/*.jpg`
    - `folder`: `ImageFolder(path)`. 상대 경로가 현재 위치에 없으면 `<data_root>/path`로 본다
    - `fake`: `FakeData` (데이터 없이 파이프라인 점검용)
    - `classes`: 주어지면 해당 label만 남긴 `Subset` (이상탐지에서 normal class만 학습할 때)
    `celeba`·`folder`·`fake`는 `train` 인자를 무시한다.
    """
    key = name.lower()
    data_root = get_data_root(root)

    if key in _TORCHVISION_DATASETS:
        tf = image_transform(image_size, channels)
        try:
            ds: Dataset = _TORCHVISION_DATASETS[key](root=str(data_root), train=train, download=download, transform=tf)
        except RuntimeError as e:
            if download:
                raise
            raise RuntimeError(
                f"{name} 데이터셋을 '{data_root}'에서 찾지 못했습니다. "
                "--download 로 내려받거나, DATA_ROOT 환경변수(또는 --data_root)를 데이터가 있는 위치로 지정하십시오."
            ) from e
    elif key == "celeba":
        ds = _image_folder(data_root / "celeba", image_size, channels)
    elif key == "folder":
        if path is None:
            raise ValueError("dataset='folder'에는 path 인자(ImageFolder 루트)가 필요합니다.")
        folder = Path(path).expanduser()
        if not folder.is_absolute() and not folder.exists():
            folder = data_root / folder
        ds = _image_folder(folder, image_size, channels)
    elif key == "fake":
        ds = datasets.FakeData(
            size=num_fake,
            image_size=(channels, image_size, image_size),
            num_classes=10,
            transform=image_transform(image_size, channels),
        )
    else:
        raise ValueError(f"알 수 없는 dataset: {name!r} (mnist | fashion_mnist | cifar10 | celeba | folder | fake)")

    if classes is not None:
        ds = _filter_classes(ds, classes)
    return ds


def _image_folder(folder: Path, image_size: int, channels: int) -> Dataset:
    """ImageFolder + center crop transform. 폴더가 없으면 원인을 알려주는 에러를 낸다."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"이미지 폴더가 없습니다: '{folder}'. ImageFolder 구조(<folder>/<subdir>/*.jpg)가 필요합니다. "
            "DATA_ROOT 환경변수 또는 --data_root를 확인하십시오."
        )
    return datasets.ImageFolder(str(folder), transform=image_transform(image_size, channels, center_crop=True))


def _filter_classes(ds: Dataset, classes: list[int]) -> Subset:
    """label이 `classes`에 속하는 샘플만 남긴다. `targets` 속성이 없으면(FakeData) 한 번 순회한다."""
    keep = {int(c) for c in classes}
    targets = getattr(ds, "targets", None)
    if targets is None:
        targets = [ds[i][1] for i in range(len(ds))]
    labels = torch.as_tensor(targets).tolist()
    indices = [i for i, t in enumerate(labels) if t in keep]
    if not indices:
        raise ValueError(f"classes={sorted(keep)}에 해당하는 샘플이 없습니다.")
    return Subset(ds, indices)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """DataLoader 생성. CUDA가 있으면 pin_memory, worker가 있으면 persistent_workers.

    기본값(shuffle=True, drop_last=True)은 학습용이다. 평가·이상탐지 scoring 루프는
    `shuffle=False, drop_last=False`를 넘겨 모든 샘플을 순서대로 한 번씩 보게 한다.
    """
    if drop_last and len(dataset) < batch_size:
        warnings.warn(
            f"데이터셋 크기({len(dataset)})가 batch_size({batch_size})보다 작아 drop_last=True면 batch가 0개입니다.",
            stacklevel=2,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# image-to-image translation (pix2pix / CycleGAN)
# 무작위성은 모두 python `random` 모듈로 뽑는다 (seed_everything과 DataLoader worker seeding이 둘 다 관리).
# ---------------------------------------------------------------------------


def _list_images(folder: Path) -> list[Path]:
    """`folder` 아래(하위 폴더 포함) 이미지 파일을 정렬해 돌려준다. 없으면 에러."""
    if not folder.is_dir():
        raise FileNotFoundError(f"폴더가 없습니다: '{folder}'. DATA_ROOT 환경변수 또는 --data_root를 확인하십시오.")
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"'{folder}'에 이미지 파일({', '.join(IMG_EXTENSIONS)})이 없습니다.")
    return paths


def _load_rgb(path: Path) -> Image.Image:
    """파일 핸들을 닫고 RGB 복사본을 돌려준다."""
    with Image.open(path) as img:
        return img.convert("RGB")


def _random_params(load_size: int, crop_size: int, flip: bool) -> tuple[int, int, bool]:
    """random crop 좌상단 (x, y)와 flip 여부. paired에서는 A·B가 이 값을 공유한다."""
    x = random.randint(0, load_size - crop_size)
    y = random.randint(0, load_size - crop_size)
    return x, y, flip and random.random() < 0.5


def _preprocess(
    img: Image.Image,
    load_size: int,
    crop_size: int,
    params: tuple[int, int, bool] | None,
) -> Tensor:
    """`params`가 있으면(train) load_size로 resize → crop → flip, 없으면 crop_size로 resize만. [-1, 1] 3ch."""
    if params is not None:
        x, y, do_flip = params
        img = TF.resize(img, [load_size, load_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        img = TF.crop(img, top=y, left=x, height=crop_size, width=crop_size)
        if do_flip:
            img = TF.hflip(img)
    else:
        img = TF.resize(img, [crop_size, crop_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return TF.normalize(TF.to_tensor(img), *_MEAN_STD_RGB)


def _check_sizes(load_size: int, crop_size: int) -> None:
    if crop_size > load_size:
        raise ValueError(f"crop_size({crop_size})가 load_size({load_size})보다 클 수 없습니다.")


class PairedImageDataset(Dataset):
    """pix2pix용. `root/phase/*.jpg` 각 파일이 [A | B]를 좌우로 붙인 이미지다.

    augment 시: A·B에 같은 resize(load_size) → random crop → flip을 적용한다 (픽셀 정렬 유지).
    augment 아닐 때: crop_size로 resize만 한다.
    `augment=None`이면 `phase == "train"`일 때만 augment한다. False면 train split도 resize만 한다.
    반환: `{"A", "B", "A_path", "B_path"}`. `direction="BtoA"`면 A·B를 바꿔 준다.
    """

    def __init__(
        self,
        root: str | Path,
        phase: str = "train",
        direction: str = "AtoB",
        load_size: int = 286,
        crop_size: int = 256,
        flip: bool = True,
        *,
        augment: bool | None = None,
    ) -> None:
        if direction not in ("AtoB", "BtoA"):
            raise ValueError(f"direction은 AtoB 또는 BtoA: {direction!r}")
        _check_sizes(load_size, crop_size)
        self.root = Path(root).expanduser()
        self.phase = phase
        self.direction = direction
        self.load_size = load_size
        self.crop_size = crop_size
        self.flip = flip
        self.augment = phase == "train" if augment is None else augment
        self.paths = _list_images(self.root / phase)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        ab = _load_rgb(path)
        w, h = ab.size
        a = ab.crop((0, 0, w // 2, h))
        b = ab.crop((w // 2, 0, w, h))
        params = _random_params(self.load_size, self.crop_size, self.flip) if self.augment else None
        img_a = _preprocess(a, self.load_size, self.crop_size, params)  # (3, crop, crop)
        img_b = _preprocess(b, self.load_size, self.crop_size, params)
        if self.direction == "BtoA":
            img_a, img_b = img_b, img_a
        return {"A": img_a, "B": img_b, "A_path": str(path), "B_path": str(path)}


class UnpairedImageDataset(Dataset):
    """CycleGAN용. `root/{phase}A`, `root/{phase}B`의 서로 짝이 없는 이미지.

    길이는 max(|A|, |B|). A는 index 순환, B는 `serial=False`면 무작위로 뽑아 짝 고정을 피한다.
    augment 시: A·B 각각 독립적으로 resize(load_size) → random crop → flip. 아닐 때: crop_size로 resize만.
    `augment=None`이면 `phase == "train"`일 때만 augment한다. False면 train split도 resize만 한다.
    반환: `{"A", "B", "A_path", "B_path"}`.
    """

    def __init__(
        self,
        root: str | Path,
        phase: str = "train",
        load_size: int = 286,
        crop_size: int = 256,
        flip: bool = True,
        serial: bool = False,
        *,
        augment: bool | None = None,
    ) -> None:
        _check_sizes(load_size, crop_size)
        self.root = Path(root).expanduser()
        self.phase = phase
        self.load_size = load_size
        self.crop_size = crop_size
        self.flip = flip
        self.serial = serial
        self.augment = phase == "train" if augment is None else augment
        self.paths_a = _list_images(self.root / f"{phase}A")
        self.paths_b = _list_images(self.root / f"{phase}B")

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, index: int) -> dict:
        path_a = self.paths_a[index % len(self.paths_a)]
        if self.serial:
            path_b = self.paths_b[index % len(self.paths_b)]
        else:
            path_b = self.paths_b[random.randint(0, len(self.paths_b) - 1)]
        params_a = _random_params(self.load_size, self.crop_size, self.flip) if self.augment else None
        params_b = _random_params(self.load_size, self.crop_size, self.flip) if self.augment else None
        img_a = _preprocess(_load_rgb(path_a), self.load_size, self.crop_size, params_a)  # (3, crop, crop)
        img_b = _preprocess(_load_rgb(path_b), self.load_size, self.crop_size, params_b)
        return {"A": img_a, "B": img_b, "A_path": str(path_a), "B_path": str(path_b)}
