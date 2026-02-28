"""
M2O 特徵轉移訓練腳本（3 類 baseline 版本）

支援三種 backbone 初始化方式：
  1. fusion        : 與原本 M2O_feature_transfer_train.py 相同，source+target 線性融合。
  2. single_source : 只用 source 模型的 state_dict 初始化 backbone，新的 3 類 head 由資料訓練。
  3. single_target : 只用 target 模型的 state_dict 初始化 backbone，新的 3 類 head 由資料訓練。

最終類別數 num_classes 完全由 feature_root 底下子資料夾數量決定：
  - feature_root/
      badnets/
      refool/
      clean/
  → num_classes = 3，logit index 對應 sorted(os.listdir(feature_root)) 的順序。
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

# 使用學姊腳本同一套模型與設定（O2M / M2O）
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
        description=(
            "Source+Target fusion / single-backbone transfer: "
            "init Transfer_Net by fusion or single source/target, "
            "fine-tune on target features (3 類以上), evaluate on target images."
        )
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
        help="target 特徵根目錄，底下包含多個子資料夾與 features_*.npy（每個子資料夾代表一個類別）。",
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
        help="（僅在 fusion 模式有效）融合時 source 權重 (預設 0.5)。",
    )
    parser.add_argument(
        "--para_target",
        type=float,
        default=0.5,
        help="（僅在 fusion 模式有效）融合時 target 權重 (預設 0.5)。",
    )
    parser.add_argument(
        "--backbone_init_mode",
        type=str,
        default="fusion",
        choices=["fusion", "single_source", "single_target", "scratch"],
        help=(
            "backbone 初始化策略："
            "'fusion' = source+target 線性融合（原 M2O）；"
            "'single_source' = 只用 source 模型初始化；"
            "'single_target' = 只用 target 模型初始化；"
            "'scratch' = 不載入任何 checkpoint，直接以隨機權重訓練 3 類模型。"
        ),
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

    - 每個子資料夾即一個類別 (badnets / refool / clean / ...)。
    - 支援 2D (N,C) 或 4D (N,C,H,W)。
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
                f"Expected 2D (N,C) or 4D (N,C,H,W) feature array in {feat_path}, "
                f"got ndim={feats.ndim} shape={feats.shape}"
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


@torch.no_grad()
def evaluate_on_images_detailed(
    model,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, dict, dict]:
    """
    測試時用影像，回傳整體準確率、各類別準確率、各類別平均信心。
    信心 = 預測類別的 softmax 機率（該類樣本上的平均）。
    """
    model.eval()
    test_flag = 1
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_conf_sum = [0.0] * num_classes

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model.predict(images, test_flag)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        pred_conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

        for i in range(labels.size(0)):
            c = labels[i].item()
            class_total[c] += 1
            if preds[i].item() == c:
                class_correct[c] += 1
            class_conf_sum[c] += pred_conf[i].item()

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_acc = total_correct / total_samples if total_samples else 0.0

    per_class_acc = {}
    per_class_conf = {}
    for c in range(num_classes):
        name = class_names[c]
        n = class_total[c]
        per_class_acc[name] = (class_correct[c] / n * 100.0) if n else 0.0
        per_class_conf[name] = (class_conf_sum[c] / n) if n else 0.0

    return overall_acc, per_class_acc, per_class_conf


@torch.no_grad()
def evaluate_2class_checkpoint(
    ckpt,
    class_names_2: List[str],
    eval_image_root: str,
    batch_size: int,
    per_digit_k: int,
    device: torch.device,
) -> Tuple[float, dict, dict]:
    """
    給定 2 類 checkpoint 與其類別名稱，建立對應的 2 類 Transfer_Net
    與 target test DataLoader，回傳 (accuracy, per_class_acc, per_class_conf)。
    """
    if len(class_names_2) != 2:
        raise ValueError(f"class_names_2 must have length 2, got {len(class_names_2)}")

    loader_2 = build_attack_balanced_test_loader(
        root=eval_image_root,
        batch_size=batch_size,
        attack_types=class_names_2,
        per_digit_k=per_digit_k,
    )

    num_classes_2 = ckpt.get("num_classes", len(ckpt.get("class_names", [])) or 2)
    model_2 = models.Transfer_Net(num_classes_2).to(device)
    model_2.load_state_dict(ckpt["state_dict"], strict=True)

    acc, per_class_acc, per_class_conf = evaluate_on_images_detailed(
        model_2, loader_2, device, class_names_2
    )
    return acc, per_class_acc, per_class_conf


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] backbone_init_mode = {args.backbone_init_mode}")

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
            f"will use target's num_classes for fusion branch."
        )
    n_class_fusion = num_classes_tgt

    # 2. 載入 target 特徵
    print(f"[INFO] Loading target features from: {args.feature_root}")
    features, labels, class_names, detected_layer = load_features_from_root(
        args.feature_root, args.feature_glob
    )
    print(f"[INFO] Target features shape = {features.shape}, classes = {class_names}")

    num_classes = len(class_names)
    n_class_cfg = CFG.get("n_class", num_classes)
    if num_classes != n_class_cfg:
        print(
            f"[WARN] Using num_classes={num_classes} (from data), "
            f"CFG n_class={n_class_cfg} overridden for this run."
        )

    # 決定 extracted_layer（維持原本邏輯）
    extracted_layer = extracted_layer_tgt or extracted_layer_src or detected_layer or "7_point"
    if not extracted_layer_tgt and not extracted_layer_src and detected_layer:
        print(f"[INFO] Using extracted_layer from feature filename: {detected_layer}")
    if extracted_layer is None:
        extracted_layer = "7_point"
        print(f"[WARN] Using default extracted_layer: {extracted_layer}")
    backbone_multi.extracted_layer = extracted_layer

    # 3. 建立最終模型 Transfer_Net(num_classes)，依據不同 init 模式初始化
    if args.backbone_init_mode == "fusion":
        # 先在 target test set 上評估原始 2 類 source/target 模型的 baseline 準確率（若可行）
        src_classes_2 = ckpt_src.get("class_names", None)
        tgt_classes_2 = ckpt_tgt.get("class_names", None)
        if (
            isinstance(src_classes_2, list)
            and isinstance(tgt_classes_2, list)
            and len(src_classes_2) == 2
            and len(tgt_classes_2) == 2
        ):
            try:
                acc_src_2, src_per_class_acc, src_per_class_conf = evaluate_2class_checkpoint(
                    ckpt=ckpt_src,
                    class_names_2=src_classes_2,
                    eval_image_root=args.eval_image_root,
                    batch_size=args.batch_size,
                    per_digit_k=args.per_digit_k,
                    device=device,
                )
                print(
                    f"[BASELINE] Source 2-class acc on target test "
                    f"({src_classes_2}) = {acc_src_2*100:.2f}%"
                )
                acc_parts = [f"{name}={src_per_class_acc[name]:.2f}%" for name in src_classes_2]
                conf_parts = [f"{name}={src_per_class_conf[name]:.4f}" for name in src_classes_2]
                print(f"  Per-class acc: {', '.join(acc_parts)}")
                print(f"  Per-class mean confidence: {', '.join(conf_parts)}")
            except Exception as e:
                print(f"[WARN] Failed to compute source 2-class baseline: {e}")

            try:
                acc_tgt_2, tgt_per_class_acc, tgt_per_class_conf = evaluate_2class_checkpoint(
                    ckpt=ckpt_tgt,
                    class_names_2=tgt_classes_2,
                    eval_image_root=args.eval_image_root,
                    batch_size=args.batch_size,
                    per_digit_k=args.per_digit_k,
                    device=device,
                )
                print(
                    f"[BASELINE] Target 2-class acc on target test "
                    f"({tgt_classes_2}) = {acc_tgt_2*100:.2f}%"
                )
                acc_parts = [f"{name}={tgt_per_class_acc[name]:.2f}%" for name in tgt_classes_2]
                conf_parts = [f"{name}={tgt_per_class_conf[name]:.4f}" for name in tgt_classes_2]
                print(f"  Per-class acc: {', '.join(acc_parts)}")
                print(f"  Per-class mean confidence: {', '.join(conf_parts)}")
            except Exception as e:
                print(f"[WARN] Failed to compute target 2-class baseline: {e}")
        else:
            print(
                "[WARN] Cannot compute 2-class baselines: "
                "ckpt_src/class_names or ckpt_tgt/class_names missing or not 2-class."
            )

        print(
            f"[INFO] Fusing Transfer_Net with para_source={args.para_source}, "
            f"para_target={args.para_target}"
        )
        # 與原本 M2O_feature_transfer_train.py 相同的融合流程
        model_1 = models.Transfer_Net(n_class_fusion)
        model_1.load_state_dict(ckpt_src["state_dict"], strict=False)
        model_1 = model_1.to(device)

        model_2 = models.Transfer_Net(n_class_fusion)
        model_2.load_state_dict(ckpt_tgt["state_dict"], strict=False)
        model_2 = model_2.to(device)

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

    elif args.backbone_init_mode in ("single_source", "single_target"):
        # single_source / single_target：只用單一模型初始化 backbone，
        # classifier_layer 的 3 類 head 由隨機初始化開始，用 3 類特徵訓練。
        if args.backbone_init_mode == "single_source":
            print("[INFO] Initializing backbone from SOURCE checkpoint only.")
            ckpt_sel = ckpt_src
        else:
            print("[INFO] Initializing backbone from TARGET checkpoint only.")
            ckpt_sel = ckpt_tgt

        src_state = ckpt_sel["state_dict"]

        model = models.Transfer_Net(num_classes)
        model = model.to(device)

        dst_state = model.state_dict()
        copied_keys = 0
        for name in dst_state.keys():
            if name in src_state and src_state[name].shape == dst_state[name].shape:
                dst_state[name] = src_state[name].clone()
                copied_keys += 1
        model.load_state_dict(dst_state, strict=False)
        print(
            f"[INFO] Copied {copied_keys} parameters from selected checkpoint into new {num_classes}-class model."
        )

    else:
        # scratch：完全不載入任何 checkpoint，隨機初始化 3 類模型
        print("[INFO] Initializing model from scratch (no pretrained checkpoint).")
        model = models.Transfer_Net(num_classes)
        model = model.to(device)

    # 4. 構建 DataLoader、Optimizer、Scheduler、Loss
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

    # 5. 訓練 + 評估
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
        acc, per_class_acc, per_class_conf = evaluate_on_images_detailed(
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
            for name in class_names:
                print(
                    f"  [{name}] acc={per_class_acc[name]:.2f}%, "
                    f"conf={per_class_conf[name]:.4f}"
                )

    # 6. 儲存模型
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
                "backbone_init_mode": args.backbone_init_mode,
            },
            args.save_model_path,
        )
        print(f"[INFO] Saved fused/single-backbone+fine-tuned target model to: {args.save_model_path}")


if __name__ == "__main__":
    main()