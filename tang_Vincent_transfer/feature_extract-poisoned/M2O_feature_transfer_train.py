"""
Source + Target 融合風格的特徵轉移訓練腳本（與學姊腳本對齊）：

  1. 載入 source 端已訓練好的 Transfer_Net (source_model_path)
  2. 載入 target baseline Transfer_Net (target_model_path)，皆學姊格式 .pth
  3. 以權重 para_source / para_target 做線性融合得到新模型（state_dict 逐參數加權）
  4. 用 target 特徵微調
  5. 在 target 影像上測試，並可將最終模型存檔

模型與訓練流程與 O2M_feature_transfer_train / target_feature_baseline_train 一致：
  - Transfer_Net，訓練時 test_flag=0，Optimizer 四組 lr，LabelSmoothingCrossEntropy，ExponentialLR(gamma=0.85)
  - 評估時 model.predict(images, test_flag=1)

使用方式範例：

    python M2O_feature_transfer_train.py \
      --source_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/source/source_badnets_clean.pth" \
      --target_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/target/target_badnets_clean.pth" \
      --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/target/Target_train_badnets_clean/" \
      --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10/test" \
      --para_source 0.9 \
      --para_target 0.1 \
      --save_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/target/M2O_target_badnets_clean.pth"
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
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

from feature_train_config import TARGET_FEATURE_CFG as TFCFG
from feature_train_config import build_attack_balanced_test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source+Target fusion transfer: fuse two Transfer_Net, fine-tune on target features, evaluate on target images."
    )
    parser.add_argument(
        "--source_model_path",
        type=str,
        required=True,
        help="source 端已訓練好分類器的 .pth 檔路徑（學姊格式）。",
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        required=True,
        help="target baseline 分類器的 .pth 檔路徑（學姊格式）。",
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
        "--para_source",
        type=float,
        default=0.5,
        help="融合時 source 權重 (預設 0.5)。",
    )
    parser.add_argument(
        "--para_target",
        type=float,
        default=0.5,
        help="融合時 target 權重 (預設 0.5)。",
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
    支援 2D (N,C) 或 4D (N,C,H,W)，與學姊腳本一致。
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


@torch.no_grad()
def evaluate_on_images(model, loader: DataLoader, device: torch.device) -> float:
    """測試時用影像，test_flag=1，完整 backbone + bottle + classifier。"""
    model.eval()
    test_flag = 1
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model.predict(images, test_flag)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    if total == 0:
        return 0.0
    return correct / total


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # 1. 載入 source 與 target baseline（學姊格式）
    print(f"[INFO] Loading source model from: {args.source_model_path}")
    ckpt_src = torch.load(args.source_model_path, map_location=device)
    num_classes_src = ckpt_src.get("num_classes", len(ckpt_src.get("class_names", [])) or 2)
    extracted_layer_src = ckpt_src.get("extracted_layer")

    print(f"[INFO] Loading target baseline model from: {args.target_model_path}")
    ckpt_tgt = torch.load(args.target_model_path, map_location=device)
    num_classes_tgt = ckpt_tgt.get("num_classes", len(ckpt_tgt.get("class_names", [])) or 2)
    extracted_layer_tgt = ckpt_tgt.get("extracted_layer")

    if num_classes_src != num_classes_tgt:
        print(
            f"[WARN] num_classes mismatch: source={num_classes_src}, target={num_classes_tgt}; "
            f"will use target's num_classes for fusion."
        )
    n_class_fusion = num_classes_tgt

    # 2. 載入 target 特徵
    print(f"[INFO] Loading target features from: {args.feature_root}")
    features, labels, class_names, detected_layer = load_features_from_root(
        args.feature_root, args.feature_glob
    )
    print(f"[INFO] Target features shape = {features.shape}, classes = {class_names}")

    num_classes = len(class_names)
    if num_classes != n_class_fusion:
        print(
            f"[WARN] fused num_classes ({n_class_fusion}) != "
            f"target subdirs ({num_classes}). Will use num_classes={num_classes} and copy matching weights."
        )

    # 決定 extracted_layer
    extracted_layer = extracted_layer_tgt or extracted_layer_src or detected_layer or "7_point"
    if not extracted_layer_tgt and not extracted_layer_src and detected_layer:
        print(f"[INFO] Using extracted_layer from feature filename: {detected_layer}")
    backbone_multi.extracted_layer = extracted_layer

    n_class = CFG.get("n_class", num_classes)
    if num_classes != n_class:
        print(
            f"[WARN] Using num_classes={num_classes} (from data), "
            f"CFG n_class={n_class} overridden for this run."
        )

    # 3. 建立兩個 Transfer_Net 並載入權重，再線性融合
    model_1 = models.Transfer_Net(n_class_fusion)
    model_1.load_state_dict(ckpt_src["state_dict"], strict=False)
    model_1 = model_1.to(device)

    model_2 = models.Transfer_Net(n_class_fusion)
    model_2.load_state_dict(ckpt_tgt["state_dict"], strict=False)
    model_2 = model_2.to(device)

    print(
        f"[INFO] Fusing Transfer_Net with para_source={args.para_source}, "
        f"para_target={args.para_target}"
    )
    model = models.Transfer_Net(num_classes)
    model = model.to(device)

    state_1 = model_1.state_dict()
    state_2 = model_2.state_dict()
    fused_state = {}
    for name in model.state_dict().keys():
        if name in state_1 and name in state_2:
            t1 = state_1[name].to(device)
            t2 = state_2[name].to(device)
            if t1.shape == t2.shape and t1.shape == model.state_dict()[name].shape:
                fused_state[name] = t1 * args.para_source + t2 * args.para_target
            else:
                fused_state[name] = model.state_dict()[name].clone()
        else:
            fused_state[name] = model.state_dict()[name].clone()
    model.load_state_dict(fused_state, strict=False)

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
        acc = evaluate_on_images(model=model, loader=image_loader, device=device)
        if acc > best_acc:
            best_acc = acc
        if (epoch - 1) % log_interval == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}/{args.epoch:03d}] "
                f"train_loss={train_loss:.6f}, eval_acc={acc*100:.2f}% (best={best_acc*100:.2f}%)"
            )

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
                "para_source": args.para_source,
                "para_target": args.para_target,
            },
            args.save_model_path,
        )
        print(f"[INFO] Saved fused+fine-tuned target model to: {args.save_model_path}")


if __name__ == "__main__":
    main()
