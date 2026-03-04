#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partition 式 Attack-Type 資料集生成腳本

將已切分好的 CIFAR-10 乾淨資料（pic_pair + pic_global）依「每個類別內 partition」
製作成 attack-type 分類資料集（clean / badnets / refool / reserve）。
僅做資料集生成與存檔，不做訓練。嚴格沿用 BackdoorBox 內 BadNets 與 Refool 的 trigger 實作。

進入虛擬環境後，cd到BackdoorBox-badnets,refool目錄下，執行指令：  source backdoorbox39_env/bin/activate
退出虛擬環境後，執行指令： deactivate

指令範例：(路徑改成自己的) 
python make_poisoned_attacktype_partition.py \
  --pair_root "/media/user906/ADATA HV620S/lab/CIFAR-10/pic_pair" \
  --global_root "/media/user906/ADATA HV620S/lab/CIFAR-10/pic_global" \
  --out_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10_v1" \
  --num_parts 5 \
  --alloc "clean=1,badnets=1,refool=1,reserve=2" \
  --reflection_root "/home/user906/2024_Soon/BackdoorBox-badnets,refool/refool_reflection_images" \
  --n_reflections 1 \
  --badnets_trigger_size 3 \
  --badnets_trigger_position random_corner \
  --badnets_trigger_value 255.0 \
  --refool_max_image_size 32 \
  --refool_ghost_rate 0.49 \
  --refool_alpha_b -1.0 \
  --refool_offset "0,0" \
  --refool_sigma -1.0 \
  --refool_ghost_alpha -1.0

各參數說明：
--pair_root：源資料集路徑
--global_root：目標資料集路徑
--out_root：輸出路徑
--num_parts：將cifar-10切成幾份(看攻擊類別與乾淨類別有多少類別，以此決定分配比例)
--alloc：分配比例
--reflection_root：反射圖路徑
--n_reflections：使用的反射圖張數
--badnets_trigger_size：BadNets trigger 邊長（格數）
--badnets_trigger_position：BadNets trigger 位置
--badnets_trigger_value：BadNets trigger 數值
--refool_max_image_size：Refool max_image_size（CIFAR-10 保持 32）
--refool_ghost_rate：Refool ghost_rate
--refool_alpha_b：Refool alpha_b
--refool_offset：Refool offset
--refool_sigma：Refool sigma
--refool_ghost_alpha：Refool ghost_alpha
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# BackdoorBox 路徑：與本腳本同專案下的 BackdoorBox-badnets,refool
_BACKDOORBOX_DIR = Path(__file__).resolve().parent.parent / "BackdoorBox-badnets,refool"
if _BACKDOORBOX_DIR.is_dir():
    sys.path.insert(0, str(_BACKDOORBOX_DIR))
else:
    raise ImportError(
        f"BackdoorBox 目錄不存在: {_BACKDOORBOX_DIR}. "
        "請將腳本放在 2024_Soon/CIFAR-10/ 並確保 BackdoorBox-badnets,refool 在 2024_Soon 下。"
    )

import torch
# NumPy 2.0 移除了 random_integers，BackdoorBox Refool 仍使用它，故在此補上相容層
if not hasattr(np.random, "random_integers"):
    np.random.random_integers = lambda a, b, size=None: np.random.randint(a, b + 1, size=size)

from core.attacks.BadNets import AddDatasetFolderTrigger
from core.attacks.Refool import AddDatasetFolderTriggerMixin, AddTriggerMixin

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
BUCKETS = ("clean", "badnets", "refool", "reserve")
SPLITS_PAIR = ("train_source", "train_target")  # 來自 pair_root
SPLITS_GLOBAL = ("val", "test")  # 來自 global_root
PREVIEW_PER_BUCKET = 5


def parse_alloc(alloc_str: str) -> Dict[str, int]:
    """解析 --alloc 'clean=1,badnets=1,refool=1,reserve=2' 為四個整數。"""
    out = {}
    for part in alloc_str.strip().split(","):
        part = part.strip()
        if "=" not in part:
            raise ValueError(f"Invalid alloc part: {part!r}. Expected key=value.")
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k not in BUCKETS:
            raise ValueError(f"Unknown bucket in alloc: {k!r}. Expected one of {BUCKETS}.")
        try:
            out[k] = int(v)
        except ValueError:
            raise ValueError(f"Alloc value must be integer: {v!r}")
    for b in BUCKETS:
        if b not in out:
            out[b] = 0
    return out


def build_badnets_pattern_weight(
    trigger_size: int,
    trigger_position: str,
    trigger_value: float,
    image_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """建 CIFAR-10 用 pattern 與 weight，預設 3x3 右下角白塊。"""
    pattern = torch.zeros((1, image_size, image_size), dtype=torch.uint8)
    weight = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    if trigger_position == "bottom_right":
        pattern[0, -trigger_size:, -trigger_size:] = int(trigger_value)
        weight[0, -trigger_size:, -trigger_size:] = 1.0
    elif trigger_position == "top_left":
        pattern[0, :trigger_size, :trigger_size] = int(trigger_value)
        weight[0, :trigger_size, :trigger_size] = 1.0
    elif trigger_position == "top_right":
        pattern[0, :trigger_size, -trigger_size:] = int(trigger_value)
        weight[0, :trigger_size, -trigger_size:] = 1.0
    elif trigger_position == "bottom_left":
        pattern[0, -trigger_size:, :trigger_size] = int(trigger_value)
        weight[0, -trigger_size:, :trigger_size] = 1.0
    else:
        raise ValueError(
            f"Unsupported trigger_position: {trigger_position}. "
            "Use one of bottom_right, top_left, top_right, bottom_left."
        )
    return pattern, weight


def get_class_image_paths(class_dir: Path) -> List[Path]:
    """取得該類別資料夾內所有圖片路徑（排序後回傳）。"""
    if not class_dir.is_dir():
        return []
    paths = []
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths, key=lambda x: x.name)


def partition_paths(
    paths: List[Path],
    num_parts: int,
    alloc: Dict[str, int],
    rng: random.Random,
) -> Dict[str, List[Path]]:
    """
    將 paths 打散後切成 num_parts 份，依 alloc 分配到四個 bucket。
    Remainder：先每份 base = n // num_parts，前 r = n % num_parts 份各多 1 張；
    份的順序對應 allocation 順序 clean -> badnets -> refool -> reserve。
    """
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    base = n // num_parts
    r = n % num_parts
    # 每「份」的 size：前 r 份為 base+1，其餘為 base
    part_sizes = [base + 1] * r + [base] * (num_parts - r)
    # 依 alloc 順序把「份」對應到 bucket
    order = []
    for b in BUCKETS:
        order.extend([b] * alloc[b])
    assert len(order) == num_parts, "alloc sum must equal num_parts"
    start = 0
    result: Dict[str, List[Path]] = {b: [] for b in BUCKETS}
    for part_idx, bucket in enumerate(order):
        size = part_sizes[part_idx]
        result[bucket].extend(paths[start : start + size])
        start += size
    return result


class RefoolApplicator(AddDatasetFolderTriggerMixin):
    """僅用於套用 Refool trigger 的 helper，繼承 BackdoorBox AddDatasetFolderTriggerMixin。"""

    def __init__(
        self,
        total_num: int,
        reflection_candidates: List[np.ndarray],
        max_image_size: int = 32,
        ghost_rate: float = 0.49,
        alpha_b: float = -1.0,
        offset: Tuple[int, int] = (0, 0),
        sigma: float = -1,
        ghost_alpha: float = -1.0,
    ):
        AddTriggerMixin.__init__(
            self,
            total_num,
            reflection_candidates,
            max_image_size=max_image_size,
            ghost_rate=ghost_rate,
            alpha_b=alpha_b,
            offset=offset,
            sigma=sigma,
            ghost_alpha=ghost_alpha,
        )


def load_reflection_candidates(reflection_root: Path, n_reflections: int) -> List[np.ndarray]:
    """從 reflection_root 用 cv2 讀取前 n_reflections 張圖。"""
    if not reflection_root.is_dir():
        raise FileNotFoundError(
            f"reflection_root 不存在或非目錄: {reflection_root}. "
            "請提供有效的 --reflection_root（含大量自然圖片的資料夾）。"
        )
    paths = []
    for p in sorted(reflection_root.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    if len(paths) == 0:
        raise FileNotFoundError(f"reflection_root 下沒有圖片: {reflection_root}")
    paths = paths[:n_reflections]
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        out.append(img)
    if len(out) == 0:
        raise ValueError(f"無法從 {reflection_root} 讀取任何圖片。")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition 式 CIFAR-10 attack-type 資料集生成（clean/badnets/refool/reserve）。"
    )
    parser.add_argument("--pair_root", type=str, required=True, help="PAIR_ROOT，含 source_train/、target_train/")
    parser.add_argument("--global_root", type=str, required=True, help="GLOBAL_ROOT，含 val/、test/（禁止使用 train）")
    parser.add_argument("--out_root", type=str, required=True, help="輸出根目錄 OUT_ROOT")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--num_parts", type=int, required=True, help="每類別切成幾份")
    parser.add_argument(
        "--alloc",
        type=str,
        required=True,
        help='例如 "clean=1,badnets=1,refool=1,reserve=2"，總和須等於 num_parts',
    )
    parser.add_argument("--reflection_root", type=str, default="", help="Refool 反射圖目錄；refool 份數>0 時必填")
    parser.add_argument("--n_reflections", type=int, default=200, help="使用的反射圖張數")
    parser.add_argument("--badnets_trigger_size", type=int, default=3, help="BadNets trigger 邊長（格數）")
    parser.add_argument(
        "--badnets_trigger_position",
        type=str,
        default="bottom_right",
        choices=("bottom_right", "top_left", "top_right", "bottom_left", "random_corner"),
        help="BadNets trigger 位置（或 random_corner：每張圖隨機落在四角之一）",
    )
    parser.add_argument("--badnets_trigger_value", type=float, default=255.0, help="BadNets trigger 數值")
    parser.add_argument("--refool_max_image_size", type=int, default=32, help="Refool max_image_size（CIFAR-10 保持 32）")
    parser.add_argument("--refool_ghost_rate", type=float, default=0.49)
    parser.add_argument("--refool_alpha_b", type=float, default=-1.0)
    parser.add_argument("--refool_offset", type=str, default="0,0", help="例如 4,4")
    parser.add_argument("--refool_sigma", type=float, default=-1.0)
    parser.add_argument("--refool_ghost_alpha", type=float, default=-1.0)
    args = parser.parse_args()

    pair_root = Path(args.pair_root).resolve()
    global_root = Path(args.global_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not pair_root.is_dir():
        raise FileNotFoundError(f"pair_root 不存在: {pair_root}. 請補上正確的 --pair_root。")
    if not global_root.is_dir():
        raise FileNotFoundError(f"global_root 不存在: {global_root}. 請補上正確的 --global_root。")

    alloc = parse_alloc(args.alloc)
    if sum(alloc[b] for b in BUCKETS) != args.num_parts:
        raise ValueError(
            f"alloc 總和 {sum(alloc[b] for b in BUCKETS)} 必須等於 num_parts={args.num_parts}. "
            f"當前 alloc: {alloc}"
        )
    if alloc["refool"] > 0 and (not args.reflection_root or not args.reflection_root.strip()):
        raise ValueError(
            "refool 份數 > 0 時必須提供 --reflection_root（含反射圖的目錄）。 "
            "請補上例如: --reflection_root /path/to/refool_reflection_images"
        )

    refool_offset: Tuple[int, int] = (0, 0)
    if args.refool_offset:
        parts = args.refool_offset.replace(" ", "").split(",")
        if len(parts) == 2:
            refool_offset = (int(parts[0]), int(parts[1]))

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 掃描四個 split 的類別與路徑
    split_dirs: Dict[str, Path] = {}
    split_dirs["train_source"] = pair_root / "source_train"
    split_dirs["train_target"] = pair_root / "target_train"
    split_dirs["val"] = global_root / "val"
    split_dirs["test"] = global_root / "test"

    for name, d in split_dirs.items():
        if not d.is_dir():
            raise FileNotFoundError(
                f"Split 目錄不存在: {d}. "
                f"pair_root 需含 source_train/、target_train/；global_root 需含 val/、test/。"
            )

    all_splits = list(SPLITS_PAIR) + list(SPLITS_GLOBAL)
    # 取得所有 split 下類別名稱（取聯集，以第一個有的為準順序）
    class_names: List[str] = []
    for split_name in all_splits:
        d = split_dirs[split_name]
        for c in sorted(d.iterdir()):
            if c.is_dir() and c.name not in class_names:
                class_names.append(c.name)
    class_names = sorted(set(class_names))
    if not class_names:
        raise FileNotFoundError("未在任何 split 下找到類別子資料夾。")

    # 建立輸出目錄：每個 split 下 clean/badnets/refool/reserve，每個下再 class
    for split_name in all_splits:
        for bucket in BUCKETS:
            for c in class_names:
                (out_root / split_name / bucket / c).mkdir(parents=True, exist_ok=True)
    (out_root / "_preview").mkdir(parents=True, exist_ok=True)
    for split_name in all_splits:
        for bucket in BUCKETS:
            (out_root / "_preview" / split_name / bucket).mkdir(parents=True, exist_ok=True)

    # BadNets trigger
    badnets_trigger = None
    if args.badnets_trigger_position != "random_corner":
        pattern, weight = build_badnets_pattern_weight(
            args.badnets_trigger_size,
            args.badnets_trigger_position,
            args.badnets_trigger_value,
        )
        badnets_trigger = AddDatasetFolderTrigger(pattern, weight)

    # Refool：載入反射圖並在 seed 已設下建立 applicator
    reflection_candidates: List[np.ndarray] = []
    refool_applicator = None
    if alloc["refool"] > 0:
        reflection_root = Path(args.reflection_root).resolve()
        reflection_candidates = load_reflection_candidates(reflection_root, args.n_reflections)
        refool_applicator = RefoolApplicator(
            total_num=500000,
            reflection_candidates=reflection_candidates,
            max_image_size=args.refool_max_image_size,
            ghost_rate=args.refool_ghost_rate,
            alpha_b=args.refool_alpha_b,
            offset=refool_offset,
            sigma=args.refool_sigma,
            ghost_alpha=args.refool_ghost_alpha,
        )

    metadata: Dict = {
        "args": {
            "pair_root": str(pair_root),
            "global_root": str(global_root),
            "out_root": str(out_root),
            "seed": args.seed,
            "num_parts": args.num_parts,
            "alloc": alloc,
            "reflection_root": args.reflection_root or None,
            "n_reflections": args.n_reflections,
            "badnets_trigger_size": args.badnets_trigger_size,
            "badnets_trigger_position": args.badnets_trigger_position,
            "badnets_trigger_value": args.badnets_trigger_value,
        },
        "splits": {},
        "path_mapping": [],
    }

    count_by_split_bucket: Dict[str, Dict[str, int]] = {s: {b: 0 for b in BUCKETS} for s in all_splits}
    refool_global_index = 0
    preview_candidates: Dict[str, Dict[str, List[Tuple[Path, Path]]]] = {
        s: {b: [] for b in BUCKETS} for s in all_splits
    }

    for split_name in all_splits:
        split_path = split_dirs[split_name]
        metadata["splits"][split_name] = {"classes": {}}

        for class_name in class_names:
            class_dir = split_path / class_name
            paths = get_class_image_paths(class_dir)
            if not paths:
                metadata["splits"][split_name]["classes"][class_name] = {
                    b: [] for b in BUCKETS
                }
                continue

            partitioned = partition_paths(paths, args.num_parts, alloc, rng)
            metadata["splits"][split_name]["classes"][class_name] = {
                b: [str(p) for p in partitioned[b]] for b in BUCKETS
            }

            for bucket in BUCKETS:
                for src_path in partitioned[bucket]:
                    out_dir = out_root / split_name / bucket / class_name
                    out_f = out_dir / src_path.name
                    if bucket == "clean" or bucket == "reserve":
                        img = cv2.imread(str(src_path))
                        if img is not None:
                            cv2.imwrite(str(out_f), img)
                            count_by_split_bucket[split_name][bucket] += 1
                            preview_candidates[split_name][bucket].append((src_path, out_f))
                    elif bucket == "badnets":
                        img = cv2.imread(str(src_path))
                        if img is not None:
                            if args.badnets_trigger_position == "random_corner":
                                corner = rng.choice(
                                    ("top_left", "top_right", "bottom_left", "bottom_right")
                                )
                                pattern, weight = build_badnets_pattern_weight(
                                    args.badnets_trigger_size,
                                    corner,
                                    args.badnets_trigger_value,
                                )
                                trigger = AddDatasetFolderTrigger(pattern, weight)
                                poisoned = trigger(img)  # returns (H,W,C) numpy
                            else:
                                poisoned = badnets_trigger(img)  # returns (H,W,C) numpy
                            cv2.imwrite(str(out_f), poisoned)
                            count_by_split_bucket[split_name][bucket] += 1
                            preview_candidates[split_name][bucket].append((src_path, out_f))
                    else:  # refool
                        img = cv2.imread(str(src_path))
                        if img is not None and refool_applicator is not None:
                            out_img = refool_applicator.add_trigger(img, refool_global_index)
                            refool_global_index += 1
                            cv2.imwrite(str(out_f), out_img)
                            count_by_split_bucket[split_name][bucket] += 1
                            preview_candidates[split_name][bucket].append((src_path, out_f))

                    metadata["path_mapping"].append({
                        "src": str(src_path),
                        "out": str(out_f),
                        "split": split_name,
                        "bucket": bucket,
                        "class": class_name,
                    })

    # Preview：每個 split 每個 bucket 隨機取 3~5 張
    preview_rng = random.Random(args.seed + 1)
    for split_name in all_splits:
        for bucket in BUCKETS:
            candidates = preview_candidates[split_name][bucket]
            n_preview = min(PREVIEW_PER_BUCKET, len(candidates))
            if n_preview > 0:
                chosen = preview_rng.sample(candidates, n_preview)
                preview_dir = out_root / "_preview" / split_name / bucket
                for i, (src_path, out_path) in enumerate(chosen):
                    if out_path.exists():
                        shutil.copy2(out_path, preview_dir / f"preview_{i}_{out_path.name}")

    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Console summary
    print("========== Partition Attack-Type Dataset Summary ==========")
    print(f"OUT_ROOT: {out_root}")
    print("Counts per split x bucket:")
    for split_name in all_splits:
        print(f"  {split_name}: clean={count_by_split_bucket[split_name]['clean']} "
              f"badnets={count_by_split_bucket[split_name]['badnets']} "
              f"refool={count_by_split_bucket[split_name]['refool']} "
              f"reserve={count_by_split_bucket[split_name]['reserve']}")
    print("Preview: OUT_ROOT/_preview/<split>/<bucket>/")
    print("Metadata: OUT_ROOT/metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
