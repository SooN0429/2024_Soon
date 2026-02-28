# 特徵抽取之預設參數（seed、batch_size、pooling 等）來自 feature_extract_config.FEATURE_EXTRACT_CFG。
# === Example commands（一次跑 attack + clean 兩套特徵）===
# Source：父目錄為 train_source，底下有 badnets / clean 等子資料夾（各含 CIFAR-10 0~9）
# 單一攻擊（舊行為，相容）
# python Extract_feature_map_v2.py \
#   --input_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10/train_source" \
#   --attack_dir badnets \
#   --clean_dir clean \
#   --output_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_" \
#   --split_name Source_train_badnets_clean \
#   --samples_per_class 100 \
#   --min_class_policy oversample \
#   --extracted_layer 7_point \
#   --pooling none
#
# 多個攻擊（例如 badnets + refool）
# python Extract_feature_map_v2.py \
#   --input_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10/train_source" \
#   --attack_dirs badnets refool \
#   --clean_dir clean \
#   --output_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_" \
#   --split_name Source_train_multiAttack_clean \
#   --samples_per_class 100 \
#   --min_class_policy oversample \
#   --extracted_layer 7_point \
#   --pooling none
#
# Target：父目錄為 train_target，同樣指定 attack_dir(s) / clean_dir
# python Extract_feature_map_v2.py \
#   --attack_dir badnets \
#   --attack_dirs badnets refool \
#   --clean_dir clean

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from feature_extract_config import (
    FEATURE_EXTRACT_CFG as ECFG,
    SOURCE_EXTRACT_PROFILE,
    TARGET_EXTRACT_PROFILE,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def str2bool(v: str) -> bool:
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def infer_domain_from_path(input_path: str) -> Optional[str]:
    """
    === Domain 推斷（infer_domain_from_path）===
    根據路徑字串自動判斷 domain 是 source 或 target。
    若都判斷不到則回傳 None，由主程式決定是否透過 --domain 覆蓋。
    """
    norm = os.path.normpath(input_path)
    parts = norm.split(os.sep)
    parts_lower = [p.lower() for p in parts]

    has_source = any("source" in p for p in parts_lower)
    has_target = any("target" in p for p in parts_lower)

    if has_source and not has_target:
        return "source"
    if has_target and not has_source:
        return "target"
    if has_source and has_target:
        # 若同時出現，依先出現者決定
        for p in parts_lower:
            if "source" in p:
                return "source"
            if "target" in p:
                return "target"
    return None


class ImageFolderWithPaths(datasets.ImageFolder):
    """擴充 ImageFolder，額外回傳影像路徑（之後可轉相對路徑存到 meta）。"""

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, target, path


class ResNet18Extractor(nn.Module):
    """
    === 抽 layer 特徵（ResNet-18 抽取中間層）===
    依據 extracted_layer 回傳學姊專案對應的 1_point ~ 9_point 特徵。
    """

    def __init__(self, extracted_layer: str, pretrained: bool = True):
        super().__init__()
        if extracted_layer not in {
            "1_point",
            "2_point",
            "3_point",
            "4_point",
            "5_point",
            "6_point",
            "7_point",
            "8_point",
            "9_point",
        }:
            raise ValueError(
                f"Unsupported extracted_layer={extracted_layer}, "
                f"must be one of 1_point, 2_point, 3_point, 4_point, 5_point, 6_point, 7_point, 8_point, 9_point."
            )

        backbone = models.resnet18(pretrained=pretrained)
        self.extracted_layer = extracted_layer

        # 拆開 backbone 的層，方便中途截斷
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1_point ~ 9_point 與舊版對齊：
        1_point: conv1 + bn1 + relu + maxpool 之後
        2_point: 上述再接 layer1[0]
        3_point: 上述再接整個 layer1
        4_point: 上述再接 layer2[0]
        5_point: 上述再接整個 layer2
        6_point: 上述再接 layer3[0]
        7_point: 上述再接整個 layer3
        8_point: 上述再接 layer4[0]
        9_point: 上述再接整個 layer4 + avgpool
        """
        # 1_point: conv1 區塊（含 maxpool）
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.extracted_layer == "1_point":
            return x

        # 2_point: 再接 layer1[0]
        x = self.layer1[0](x)
        if self.extracted_layer == "2_point":
            return x

        # 3_point: 再接整個 layer1（剩餘 block）
        x = self.layer1[1:](x)
        if self.extracted_layer == "3_point":
            return x

        # 4_point: 再接 layer2[0]
        x = self.layer2[0](x)
        if self.extracted_layer == "4_point":
            return x

        # 5_point: 再接整個 layer2（剩餘 block）
        x = self.layer2[1:](x)
        if self.extracted_layer == "5_point":
            return x

        # 6_point: 再接 layer3[0]
        x = self.layer3[0](x)
        if self.extracted_layer == "6_point":
            return x

        # 7_point: 再接整個 layer3（剩餘 block）
        x = self.layer3[1:](x)
        if self.extracted_layer == "7_point":
            return x

        # 8_point: 再接 layer4[0]
        x = self.layer4[0](x)
        if self.extracted_layer == "8_point":
            return x

        # 9_point: 再接整個 layer4（剩餘 block）+ avgpool
        x = self.layer4[1:](x)
        x = self.avgpool(x)
        if self.extracted_layer == "9_point":
            return x

        # 理論上不會到這裡
        raise RuntimeError(f"Unexpected extracted_layer={self.extracted_layer}")


def build_transform(transform_mode: str) -> transforms.Compose:
    """
    Transform 選擇：
    - safe_eval: deterministic，Resize + CenterCrop + ToTensor + Normalize
    - train_aug: 類似舊 data_loader 的 train（含隨機增強）
    """
    if transform_mode == "safe_eval":
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif transform_mode == "train_aug":
        return transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        raise ValueError(f"Unsupported transform_mode={transform_mode}")


def balanced_sample_indices(
    dataset: ImageFolderWithPaths,
    samples_per_class: int,
    min_class_policy: str,
    seed: int,
    max_total_samples: Optional[int] = None,
) -> Tuple[List[int], Dict[int, int]]:
    """
    === 等量抽樣：每類 K 張，依 min_class_policy ===
    回傳被選中的 sample indices，以及每個 class 實際抽樣數。
    """
    rng = random.Random(seed)

    # 依 label 分組
    indices_by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(idx)

    selected_indices: List[int] = []
    per_class_counts: Dict[int, int] = {}

    for cls, indices in indices_by_class.items():
        n = len(indices)
        k = samples_per_class

        if n >= k:
            chosen = rng.sample(indices, k)
        else:
            if min_class_policy == "truncate":
                chosen = list(indices)
            elif min_class_policy == "error":
                # 找回 class 名稱，方便錯誤訊息
                class_name = None
                for name, idx_val in dataset.class_to_idx.items():
                    if idx_val == cls:
                        class_name = name
                        break
                raise ValueError(
                    f"class '{class_name}' (idx={cls}) has only {n} samples < K={k} "
                    f"with min_class_policy='error'."
                )
            elif min_class_policy == "oversample":
                if n == 0:
                    class_name = None
                    for name, idx_val in dataset.class_to_idx.items():
                        if idx_val == cls:
                            class_name = name
                            break
                    raise ValueError(
                        f"class '{class_name}' (idx={cls}) has 0 samples, cannot oversample."
                    )
                chosen = [indices[rng.randrange(n)] for _ in range(k)]
            else:
                raise ValueError(f"Unsupported min_class_policy={min_class_policy}")

        per_class_counts[cls] = len(chosen)
        selected_indices.extend(chosen)

    # 如有 max_total_samples，整體再截斷
    if max_total_samples is not None and len(selected_indices) > max_total_samples:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:max_total_samples]

        # 重新計算截斷後的各類數量
        per_class_counts = defaultdict(int)
        for idx in selected_indices:
            _, label = dataset.samples[idx]
            per_class_counts[label] += 1

    return selected_indices, dict(per_class_counts)


def _run_one_subset(
    args,
    data_root: str,
    subset_name: str,
    dataset_role: str,
    output_dir: str,
    domain: str,
    device: torch.device,
    extractor: nn.Module,
    transform: transforms.Compose,
) -> None:
    """
    對單一子資料夾（attack 或 clean）做等量抽樣、抽特徵、存 npy + meta。
    subset_name 用於輸出子目錄與 meta（例如 badnets / clean）。
    """
    dataset = ImageFolderWithPaths(root=data_root, transform=transform)
    class_to_idx = dataset.class_to_idx

    # === 等量抽樣：每類 K 張，依 min_class_policy ===
    selected_indices, per_class_counts = balanced_sample_indices(
        dataset=dataset,
        samples_per_class=args.samples_per_class,
        min_class_policy=args.min_class_policy,
        seed=args.seed,
        max_total_samples=args.max_total_samples,
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"[INFO] [{subset_name}] Per-class sampled counts (CIFAR-10 0~9):")
    for cls_idx, count in sorted(per_class_counts.items(), key=lambda x: x[0]):
        cls_name = idx_to_class.get(cls_idx, str(cls_idx))
        print(f"  class {cls_idx} ('{cls_name}'): {count}")

    subset = Subset(dataset, selected_indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    all_features: List[np.ndarray] = []
    all_relpaths: List[str] = []

    # === 抽 layer 特徵（ResNet-18 抽取中間層）===
    with torch.no_grad():
        for batch in dataloader:
            images, _, paths = batch
            images = images.to(device)
            feats = extractor(images)

            if args.pooling == "avg":
                if feats.ndim == 4:
                    feats = feats.mean(dim=(2, 3))
                elif feats.ndim == 2:
                    pass
            elif args.pooling == "none":
                pass
            else:
                raise ValueError(f"Unsupported pooling={args.pooling}")

            feats_np = feats.detach().cpu().numpy()
            all_features.append(feats_np)

            if args.save_filenames:
                for p in paths:
                    rel = os.path.relpath(p, data_root)
                    all_relpaths.append(rel)

    if not all_features:
        raise RuntimeError(f"No features extracted for {subset_name}; check dataset and sampling.")

    features_array = np.concatenate(all_features, axis=0)
    n_samples = features_array.shape[0]
    # 每個樣本同一類別 label，直接使用字串（如 'clean'、'badnets'、'refool'）
    labels_array = np.full((n_samples,), subset_name, dtype=object)

    print(f"[INFO] [{subset_name}] feature array shape = {features_array.shape}")
    print(f"[INFO] [{subset_name}] labels array shape  = {labels_array.shape} (label string = '{subset_name}')")

    # === 存 npy + meta 資訊 ===
    base_name = f"{args.backbone}_{args.extracted_layer}_K{args.samples_per_class}"
    features_path = os.path.join(output_dir, f"features_{base_name}.npy")
    labels_path = os.path.join(output_dir, f"labels_{base_name}.npy")
    meta_path = os.path.join(output_dir, f"meta_{base_name}.json")

    os.makedirs(output_dir, exist_ok=True)
    np.save(features_path, features_array)
    np.save(labels_path, labels_array)

    meta: Dict = {
        "dataset_role": dataset_role,
        "subset_name": subset_name,
        "domain": domain,
        "split_name": args.split_name,
        "data_root": data_root,
        "output_dir": output_dir,
        "backbone": args.backbone,
        "extracted_layer": args.extracted_layer,
        "pooling": args.pooling,
        "label_semantics": "Saved labels in labels_*.npy are category strings (see saved_label_value). class_to_idx above is CIFAR-10 structure for balanced sampling only.",
        "saved_label_value": subset_name,
        "samples_per_class_requested": args.samples_per_class,
        "samples_per_class_actual": {
            idx_to_class.get(cls_idx, str(cls_idx)): int(count)
            for cls_idx, count in per_class_counts.items()
        },
        "num_samples": int(features_array.shape[0]),
        "feature_shape": list(features_array.shape),
        "class_to_idx": class_to_idx,
        "params": {
            "seed": args.seed,
            "min_class_policy": args.min_class_policy,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "pretrained": bool(args.pretrained),
            "transform_mode": args.transform_mode,
            "max_total_samples": args.max_total_samples,
            "save_filenames": bool(args.save_filenames),
        },
    }
    if args.save_filenames:
        meta["filenames"] = all_relpaths

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[INFO] [{subset_name}] Saved features to: {features_path}")
    print(f"[INFO] [{subset_name}] Saved labels  to: {labels_path}")
    print(f"[INFO] [{subset_name}] Saved meta    to: {meta_path}")


def extract_features(args):
    # 設定隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.input_root is None:
        raise ValueError(
            "input_root 未指定。請在 feature_extract_config.py 的 FEATURE_EXTRACT_CFG 中設定 input_root，"
            "或於命令列傳入 --input_root。"
        )
    if args.domain not in ("source", "target"):
        raise ValueError(f"domain 必須為 source 或 target，目前為 {args.domain!r}。")

    domain = args.domain
    subdir = "train_source" if domain == "source" else "train_target"
    input_root_abs = os.path.abspath(os.path.join(args.input_root, subdir))
    if not os.path.isdir(input_root_abs):
        raise ValueError(f"目錄不存在: {input_root_abs}（請確認 {args.input_root} 底下有 {subdir}）。")

    if getattr(args, "attack_dirs", None):
        attack_names = list(dict.fromkeys(args.attack_dirs))
    else:
        attack_names = [args.attack_dir] if args.attack_dir else []

    if not attack_names:
        raise ValueError("attack_dirs 或 attack_dir 未設定，且 profile 中無 attack_dirs。")

    clean_root = os.path.join(input_root_abs, args.clean_dir)
    if not os.path.isdir(clean_root):
        raise ValueError(f"Expected directory not found: {clean_root} (for clean)")

    attack_roots = []
    for attack_name in attack_names:
        attack_root = os.path.join(input_root_abs, attack_name)
        if not os.path.isdir(attack_root):
            raise ValueError(f"Expected directory not found: {attack_root} (for attack '{attack_name}')")
        attack_roots.append((attack_name, attack_root))

    if args.output_root is None:
        raise ValueError(
            "output_root 未指定。請在 feature_extract_config.py 的 FEATURE_EXTRACT_CFG 中設定 output_root，"
            "或於命令列傳入 --output_root。"
        )
    output_root = args.output_root

    if args.split_name is None:
        Domain = "Source" if domain == "source" else "Target"
        n_class = len(attack_names) + 1
        class_part = "_".join(attack_names + [args.clean_dir])
        args.split_name = f"{Domain}_train_{n_class}class({class_part})"
    split_name = args.split_name

    base_output = os.path.join(output_root, domain, split_name)

    print(f"[INFO] input_root (effective) = {input_root_abs}")
    print(f"[INFO] attack_dirs = {attack_names}")
    print(f"[INFO] clean_dir  = {args.clean_dir} -> {clean_root}")
    print(f"[INFO] domain = {domain}")
    print(f"[INFO] split_name = {split_name}")
    print(f"[INFO] base_output = {base_output}")

    transform = build_transform(args.transform_mode)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] Using device = {device}")

    if args.backbone != "resnet18":
        raise ValueError(f"Only resnet18 is supported currently, got {args.backbone}.")

    extractor = ResNet18Extractor(extracted_layer=args.extracted_layer, pretrained=args.pretrained)
    extractor.to(device)
    extractor.eval()

    # 先抽多個攻擊類別，再抽乾淨類別
    for attack_name, attack_root in attack_roots:
        output_dir_attack = os.path.join(base_output, attack_name)
        _run_one_subset(
            args,
            data_root=attack_root,
            subset_name=attack_name,
            dataset_role="attack",
            output_dir=output_dir_attack,
            domain=domain,
            device=device,
            extractor=extractor,
            transform=transform,
        )

    output_dir_clean = os.path.join(base_output, args.clean_dir)
    _run_one_subset(
        args,
        data_root=clean_root,
        subset_name=args.clean_dir,
        dataset_role="clean",
        output_dir=output_dir_clean,
        domain=domain,
        device=device,
        extractor=extractor,
        transform=transform,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="抽取 ImageFolder 資料集之 ResNet-18 中間層特徵，並儲存成 .npy + meta.json。"
    )

    parser.add_argument(
        "--input_root",
        type=str,
        default=ECFG.get("input_root"),
        help="父目錄路徑，底下需有 train_source 與 train_target；程式依 --domain 使用其一。未指定時使用 feature_extract_config 的 input_root。",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["source", "target"],
        default=None,
        help="來源端或目標端，必填；決定使用 SOURCE_EXTRACT_PROFILE 或 TARGET_EXTRACT_PROFILE，並進入 train_source 或 train_target。未指定時會報錯。",
    )
    parser.add_argument(
        "--attack_dir",
        type=str,
        default=None,
        help="單一攻擊類別子資料夾名稱。未指定時由 profile 的 attack_dirs 第一項填入。",
    )
    parser.add_argument(
        "--attack_dirs",
        type=str,
        nargs="+",
        default=None,
        help="多個攻擊類別子資料夾名稱。未指定時使用 profile 的 attack_dirs。",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=None,
        help="乾淨類別子資料夾名稱。未指定時使用 profile 的 clean_dir。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=ECFG.get("output_root"),
        help="features 輸出根目錄。未指定時使用 feature_extract_config 的 output_root；若 config 也未設定則報錯並提醒使用者。",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="未指定時依 domain、attack_dirs、clean_dir 自動產生（格式：Domain_train_nclass(類別名)）；若提供則覆蓋自動值。",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="每類欲抽樣張數 K。未指定時使用 profile 的 samples_per_class。",
    )
    parser.add_argument(
        "--min_class_policy",
        type=str,
        choices=["truncate", "error", "oversample"],
        default=ECFG["min_class_policy"],
        help="當某類少於 K 張時的策略：truncate/error/oversample。",
    )
    parser.add_argument("--seed", type=int, default=ECFG["seed"], help="隨機種子（抽樣與 oversample 用）。")
    parser.add_argument("--batch_size", type=int, default=ECFG["batch_size"], help="特徵抽取的 batch size。")
    parser.add_argument("--num_workers", type=int, default=ECFG["num_workers"], help="DataLoader workers 數量。")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=ECFG["device"] if (ECFG["device"] == "cuda" and torch.cuda.is_available()) else "cpu",
        help="裝置：cpu 或 cuda。",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18"],
        default=ECFG["backbone"],
        help="backbone 模型，目前僅支援 resnet18。",
    )
    parser.add_argument(
        "--pretrained",
        type=str2bool,
        default=ECFG["pretrained"],
        help="是否使用 ImageNet 預訓練權重（true/false）。",
    )
    parser.add_argument(
        "--extracted_layer",
        type=str,
        default=None,
        choices=[
            "1_point",
            "2_point",
            "3_point",
            "4_point",
            "5_point",
            "6_point",
            "7_point",
            "8_point",
            "9_point",
        ],
        help="要抽取的層名稱：1_point ~ 9_point。未指定時使用 profile 的 extracted_layer。",
    )
    parser.add_argument(
        "--save_filenames",
        type=str2bool,
        default=ECFG["save_filenames"],
        help="是否在 meta 中存每張圖的相對檔名（true/false）。",
    )
    parser.add_argument(
        "--transform_mode",
        type=str,
        choices=["safe_eval", "train_aug"],
        default=ECFG["transform_mode"],
        help="transform 模式：safe_eval（Resize+CenterCrop+Normalize）或 train_aug（隨機增強）。",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["none", "avg"],
        default=ECFG["pooling"],
        help="none：輸出 CxHxW；avg：global average pooling 成 NxC。",
    )
    parser.add_argument(
        "--max_total_samples",
        type=int,
        default=ECFG["max_total_samples"],
        help="限制總抽樣數（debug 用，可為 None 表示不限制）。",
    )

    return parser.parse_args()


def apply_domain_profile(args):
    """
    依 args.domain 選用 SOURCE_EXTRACT_PROFILE 或 TARGET_EXTRACT_PROFILE，
    將未由 CLI 指定的參數補上 profile 預設值。split_name 不在此填入，由 extract_features 依 attack_dirs、clean_dir 自動產生。
    """
    if args.domain is None:
        raise ValueError(
            "請指定 --domain source 或 --domain target。"
        )
    if args.domain not in ("source", "target"):
        raise ValueError(f"domain 必須為 source 或 target，目前為 {args.domain!r}。")

    profile = SOURCE_EXTRACT_PROFILE if args.domain == "source" else TARGET_EXTRACT_PROFILE

    if getattr(args, "samples_per_class", None) is None:
        args.samples_per_class = profile["samples_per_class"]
    if getattr(args, "extracted_layer", None) is None:
        args.extracted_layer = profile["extracted_layer"]
    if getattr(args, "clean_dir", None) is None:
        args.clean_dir = profile["clean_dir"]
    if getattr(args, "attack_dirs", None) is None or (
        isinstance(getattr(args, "attack_dirs", None), list) and len(args.attack_dirs) == 0
    ):
        args.attack_dirs = list(profile["attack_dirs"])
    if getattr(args, "attack_dir", None) is None:
        args.attack_dir = args.attack_dirs[0] if args.attack_dirs else None

    if args.samples_per_class is None or args.extracted_layer is None:
        raise ValueError("samples_per_class 與 extracted_layer 為必填，請在 profile 或 CLI 中設定。")


def main():
    args = parse_args()
    apply_domain_profile(args)
    extract_features(args)


if __name__ == "__main__":
    main()


