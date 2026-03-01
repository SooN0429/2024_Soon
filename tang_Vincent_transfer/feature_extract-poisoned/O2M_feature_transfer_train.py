"""
O2M 風格的特徵轉移訓練腳本（與學姊腳本對齊）：

  1. 載入 source 端已訓練好的 Transfer_Net（學姊格式 .pth，由 source_feature_model_train.py 輸出）
  2. 用 target 端萃取好的特徵 (features_*.npy) 進行微調
  3. 在 target 影像上測試，並可將微調後的模型存檔

模型與訓練流程與 source_feature_model_train / target_feature_baseline_train 一致：
  - Transfer_Net（base_network + bottle_layer + classifier_layer）
  - 訓練時 test_flag=0，Optimizer 四組 lr，LabelSmoothingCrossEntropy，ExponentialLR(gamma=0.85)
  - 評估時 model.predict(images, test_flag=1)

使用方式範例：

    python O2M_feature_transfer_train.py \
      --source_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/source/source_badnets_clean.pth" \
      --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/target/Target_train_badnets_clean/" \
      --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10/test" \
      --save_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/target/O2M_target_badnets_clean.pth"
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

# 使用學姊腳本同一套模型與設定（O2M）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_O2M_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "O2M"))
if _O2M_DIR not in sys.path:
    sys.path.insert(0, _O2M_DIR)

import backbone_multi
import call_resnet18_multi  # noqa: F401 - 供 models 使用
import models
from config import CFG
import LabelSmoothing as LS
import utils  # noqa: F401 - CFG/log 可能引用

from eval_utils import evaluate_on_images_with_per_class
from feature_train_config import TARGET_FEATURE_CFG as TFCFG
from feature_train_config import build_attack_balanced_test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="O2M-style transfer: load source Transfer_Net, fine-tune on target features, evaluate on test images."
    )
    parser.add_argument(
        "--source_model_path",
        type=str,
        required=True,
        help="source 端已訓練好分類器的 .pth 檔路徑 (學姊格式，由 source_feature_model_train.py 輸出)。",
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        required=True,
        help="target 特徵根目錄，底下包含多個子資料夾與 features_*.npy。",
    )
    parser.add_argument(
        "--feature_glob",
        type=str,
        default="features_*.npy",
        help="在每個子資料夾底下尋找特徵檔的樣式 (預設: features_*.npy)。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TFCFG["batch_size"],
        help="訓練與測試 batch size（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TFCFG["lr"],
        help="微調的學習率。",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=TFCFG["epoch"],
        help="訓練 epoch 數。",
    )
    parser.add_argument(
        "--eval_image_root",
        type=str,
        required=True,
        help="target 影像測試集根目錄 (例如 poisoned_Cifar-10/test)。",
    )
    parser.add_argument(
        "--per_digit_k",
        type=int,
        default=TFCFG["per_digit_k"],
        help="每個 digit(0-9) 在每個攻擊型態資料夾中抽取的最大樣本數。",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="微調後模型儲存路徑 (.pth)。若不指定則不存檔。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="運算裝置，cuda 或 cpu。",
    )
    return parser.parse_args()


def load_features_from_root(
    feature_root: str,
    feature_glob: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
    """
    掃描 feature_root 底下的每個子資料夾，讀 features_*.npy，
    回傳 (features, labels, class_names, detected_layer)。
    支援 2D (N,C) 或 4D (N,C,H,W)，與學姊 source_feature_model_train 一致。
    """
    class_dirs = [
        d for d in sorted(os.listdir(feature_root))
        if os.path.isdir(os.path.join(feature_root, d))
    ]
    if not class_dirs:
        raise RuntimeError(f"No subdirectories found under feature_root={feature_root}")

    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    detected_layer: Optional[str] = None

    for class_idx, class_name in enumerate(class_dirs):
        subdir = os.path.join(feature_root, class_name)
        pattern = os.path.join(subdir, feature_glob)
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise RuntimeError(f"No feature files matching {feature_glob} in {subdir}")
        feat_path = candidates[0]
        feats = np.load(feat_path)
        if feats.ndim not in (2, 4):
            raise RuntimeError(
                f"Expected 2D (N,C) or 4D (N,C,H,W) feature array in {feat_path}, got ndim={feats.ndim} shape={feats.shape}"
            )
        labels = np.full((feats.shape[0],), class_idx, dtype=np.int64)
        all_feats.append(feats)
        all_labels.append(labels)

        if detected_layer is None:
            basename = os.path.basename(feat_path)
            m = re.search(r"(\d+_point)", basename)
            if m:
                detected_layer = m.group(1)

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, class_dirs, detected_layer


def build_feature_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> DataLoader:
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    return loader


def train_one_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """與學姊相同：test_flag=0。4D (B,C,H,W) 直接送入；2D (B,C) 則 reshape 成 (B,C,1,1)。"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    test_flag = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C) -> (B, C, 1, 1)
        y = torch.squeeze(y)
        optimizer.zero_grad()
        _, source_clf = model(x, y, test_flag)
        loss = criterion(source_clf, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # 1. 載入 source 模型（學姊格式：state_dict, extracted_layer, num_classes）
    print(f"[INFO] Loading source model from: {args.source_model_path}")
    ckpt = torch.load(args.source_model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    extracted_layer_ckpt = ckpt.get("extracted_layer")
    num_classes_src = ckpt.get("num_classes", len(ckpt.get("class_names", [])) or 2)

    # 2. 載入 target 特徵
    print(f"[INFO] Loading target features from: {args.feature_root}")
    features, labels, class_names, detected_layer = load_features_from_root(
        args.feature_root, args.feature_glob
    )
    print(f"[INFO] Target features shape = {features.shape}, classes = {class_names}")

    num_classes = len(class_names)
    if num_classes_src != num_classes:
        print(
            f"[WARN] source num_classes ({num_classes_src}) != "
            f"target subdirs ({num_classes}). "
            f"Will use num_classes={num_classes} and copy matching weights."
        )

    # 決定 extracted_layer：checkpoint > 檔名解析 > 預設
    if extracted_layer_ckpt is not None:
        extracted_layer = extracted_layer_ckpt
    elif detected_layer is not None:
        extracted_layer = detected_layer
        print(f"[INFO] Using extracted_layer from feature filename: {extracted_layer}")
    else:
        extracted_layer = "7_point"
        print(f"[WARN] Using default extracted_layer: {extracted_layer}")

    backbone_multi.extracted_layer = extracted_layer

    n_class = CFG.get("n_class", num_classes)
    if num_classes != n_class:
        print(
            f"[WARN] Using num_classes={num_classes} (from data), "
            f"CFG n_class={n_class} overridden for this run."
        )

    # 3. 建立模型：Transfer_Net，並載入 source 權重（若類別數不同則只複製形狀一致的 key）
    model = models.Transfer_Net(num_classes)
    model = model.to(device)

    dst_state = model.state_dict()
    for name in dst_state.keys():
        if name in state_dict and state_dict[name].shape == dst_state[name].shape:
            dst_state[name] = state_dict[name].clone()
    model.load_state_dict(dst_state, strict=False)

    train_loader = build_feature_dataloader(features, labels, args.batch_size)

    optimizer = torch.optim.Adam(
        [
            {"params": model.base_network.parameters(), "lr": 100 * args.lr},
            {"params": model.base_network.avgpool.parameters(), "lr": 100 * args.lr},
            {"params": model.bottle_layer.parameters(), "lr": 10 * args.lr},
            {"params": model.classifier_layer.parameters(), "lr": 10 * args.lr},
        ],
        lr=args.lr,
        betas=CFG["betas"],
        weight_decay=CFG["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.85, verbose=False
    )
    criterion = LS.LabelSmoothingCrossEntropy(reduction="sum")

    print(f"[INFO] Building target image DataLoader from: {args.eval_image_root}")
    attack_types = class_names
    image_loader = build_attack_balanced_test_loader(
        root=args.eval_image_root,
        batch_size=args.batch_size,
        attack_types=attack_types,
        per_digit_k=args.per_digit_k,
    )

    best_acc = 0.0
    log_interval = TFCFG["log_interval"]
    for epoch in range(1, args.epoch + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()
        acc, details = evaluate_on_images_with_per_class(
            model=model,
            loader=image_loader,
            device=device,
            class_names=class_names,
        )
        if acc > best_acc:
            best_acc = acc
        if (epoch - 1) % log_interval == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}/{args.epoch:03d}] "
                f"train_loss={train_loss:.6f}, eval_acc={acc*100:.2f}% (best={best_acc*100:.2f}%)"
            )
            if details:
                acc_parts = [f"{name}={details['per_class_acc'][name]*100:.2f}%" for name in details["per_class_acc"]]
                conf_parts = [f"{name}={details['per_class_confidence'][name]:.3f}" for name in details["per_class_confidence"]]
                print(f"  Per-class acc: {', '.join(acc_parts)}")
                print(f"  Per-class mean confidence: {', '.join(conf_parts)}")
                tp_val, fp_val = details.get("tp_confidence", float("nan")), details.get("fp_confidence", float("nan"))
                tp_str = f"{tp_val:.3f}" if tp_val == tp_val else "N/A"
                fp_str = f"{fp_val:.3f}" if fp_val == fp_val else "N/A"
                print(f"  TP Confidence (correct): {tp_str}, FP Confidence (wrong): {fp_str}")

    if args.save_model_path:
        save_dir = os.path.dirname(args.save_model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "extracted_layer": extracted_layer,
                "num_classes": num_classes,
            },
            args.save_model_path,
        )
        print(f"[INFO] Saved transferred target model to: {args.save_model_path}")


if __name__ == "__main__":
    main()
