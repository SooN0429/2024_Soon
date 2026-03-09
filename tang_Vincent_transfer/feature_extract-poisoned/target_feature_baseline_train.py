"""
用 target 端已萃取的特徵 (features_*.npy) 訓練二分類模型，設定與 source_feature_model_train.py 一致。
在指定的影像資料集上測試，用來當作之後轉移學習 (source→target、source+target 融合) 的 baseline。

訓練方式與 source 相同：
  - 使用 Transfer_Net（base_network + bottle_layer + classifier_layer）
  - 訓練時輸入為預萃取特徵，forward 時 test_flag=0，只跑 backbone 後段 + bottle + classifier
  - Optimizer：base_network、avgpool、bottle_layer、classifier_layer（四組 lr）
  - Loss：LabelSmoothingCrossEntropy；Scheduler：ExponentialLR(gamma=0.85)
  - 支援 2D (N,C) 或 4D (N,C,H,W) 特徵格式

本版腳本進一步自動化參數（對齊 source_feature_model_train.py）：
  - extracted_layer：若未指定，從 features_*.npy 檔名自動解析 (例如 features_resnet18_7_point_K100.npy → 7_point)
  - num_classes：由 feature_root 底下子資料夾數量自動決定
  - attack_types：由 feature_root 子資料夾名稱自動決定，確保訓練 / 測試類別一致
  - save_model_path：指定訓練模型根目錄 (例如 .../trained_model_cpt)，實際檔案會存為 trained_model_cpt/target/target_<class1>_<class2>.pth

依賴 O2M 模組：models, config, backbone_multi, call_resnet18_multi, LabelSmoothing，與 CL_MAL-main 中對應的 models 系列。

使用方式範例：

    python target_feature_baseline_train.py \
      --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/target/Target_train_2class(badnets_clean)/" \
      --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10_v1/test" \
      --save_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/model1"\
      --model_class models1

features_dir 結構應與 source 類似，例如：

    target_features_root/
      badnets/
        features_resnet18_7_point_K100.npy
      clean/
        features_resnet18_7_point_K100.npy
"""

import argparse
import glob
import importlib
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms

# 使用與 source_feature_model_train.py 同一套模型與設定（O2M），並支援學長 CL_MAL-main 的 models 系列
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CL_MAL_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "CL_MAL-main"))
_O2M_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "O2M"))
if _CL_MAL_DIR not in sys.path:
    # 優先使用 CL_MAL-main 的 models* 定義
    sys.path.insert(0, _CL_MAL_DIR)
if _O2M_DIR not in sys.path:
    sys.path.insert(1, _O2M_DIR)

import backbone_multi
import call_resnet18_multi  # noqa: F401 - 供 models 使用
import models
from config import CFG
import LabelSmoothing as LS
import utils  # noqa: F401 - CFG/log 可能引用

from attack_test_dataset import AttackTypeBalancedTestDataset
from eval_utils import evaluate_on_images_with_per_class
from feature_train_config import TARGET_FEATURE_CFG as TFCFG
from feature_train_config import build_attack_balanced_test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train target baseline on extracted features (same setup as source_feature_model_train.py)."
    )
    parser.add_argument(
        "--extracted_layer",
        type=str,
        default=None,
        help="特徵萃取層，需與 .npy 特徵對應 (學姊用 5_point~8_point)。若不指定，將從 features_*.npy 檔名自動解析。",
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        required=True,
        help="目錄，底下包含多個子資料夾 (badnets, clean, ...) 與 features_*.npy。",
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
        help="學習率（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=TFCFG["epoch"],
        help="訓練 epoch 數（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--eval_image_root",
        type=str,
        required=True,
        help="用於測試的 target 影像資料集根目錄 (例如 poisoned_Cifar-10/test)。",
    )
    parser.add_argument(
        "--per_digit_k",
        type=int,
        default=TFCFG["per_digit_k"],
        help="每個 digit(0-9) 在每個攻擊型態資料夾中抽取的最大樣本數（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="訓練完成後模型儲存根目錄 (例如 .../trained_model_cpt)。若不指定則不存檔。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="運算裝置，cuda 或 cpu。",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="models",
        help=(
            "用於建構 Transfer_Net 的模組名，與 model3class_fusion2025 的 "
            "--source_model_class / --target_model_class 對應。"
            " 常見：models、models1、models1_1、models1_2、models2、models2_1、models2_2、"
            "models3、models3_1、models3_2、models4、models4_1、models4_2。目標端做異構融合時可選擇與來源不同的架構。"
        ),
    )
    return parser.parse_args()


def load_features_from_root(
    feature_root: str,
    feature_glob: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
    """
    掃描 feature_root 底下的每個子資料夾，讀 features_*.npy，
    回傳 (features, labels, class_names)。
    支援 2D (N, C) 或 4D (N, C, H, W)，與 source_feature_model_train 一致。
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

        # 嘗試從第一個特徵檔名解析 extracted_layer (例如 7_point)
        if detected_layer is None:
            basename = os.path.basename(feat_path)
            m = re.search(r"(\d+_point)", basename)
            if m:
                detected_layer = m.group(1)

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, class_dirs, detected_layer


def _infer_domain_for_target() -> str:
    """
    目前此腳本專門訓練 target baseline，domain 固定為 'target'。
    抽成函式純為與 source_feature_model_train 的寫法對齊。
    """
    return "target"


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
    """與 source 相同：test_flag=0。4D (B,C,H,W) 直接送入；2D (B,C) 則 reshape 成 (B,C,1,1)。"""
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

    print(f"[INFO] Loading target features from: {args.feature_root}")
    features, labels, class_names, detected_layer = load_features_from_root(
        args.feature_root, args.feature_glob
    )
    print(f"[INFO] Loaded features shape = {features.shape}, classes = {class_names}")

    # 決定實際使用的 extracted_layer：CLI > 檔名自動解析 > fallback
    if args.extracted_layer is not None:
        extracted_layer = args.extracted_layer
    elif detected_layer is not None:
        extracted_layer = detected_layer
        print(f"[INFO] Auto-detected extracted_layer from feature filename: {extracted_layer}")
    else:
        extracted_layer = "7_point"
        print(f"[WARN] Could not detect extracted_layer from filename, fallback to default: {extracted_layer}")

    backbone_multi.extracted_layer = extracted_layer

    # 分類類別數一律由 feature_root 子資料夾數決定
    num_classes = len(class_names)

    n_class = CFG.get("n_class", num_classes)
    if num_classes != n_class:
        print(
            f"[WARN] Using num_classes={num_classes} (from data), "
            f"CFG n_class={n_class} overridden for this run."
        )

    # 依 args.model_class 動態載入模組並取得 Transfer_Net（優先使用 CL_MAL-main，其次 O2M；與 model3class_fusion2025 的 --target_model_class 對應）
    mod = importlib.import_module(args.model_class)
    Transfer_Net = mod.Transfer_Net
    print(f"[INFO] model_class = {args.model_class}")

    train_loader = build_feature_dataloader(features, labels, args.batch_size)

    model = Transfer_Net(n_class) # 
    model = model.to(device)

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
    attack_types = class_names  # 測試時 attack_types 與訓練特徵類別一致
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
                # 各類別 TP/FP 平均信心
                per_tp_conf = details.get("per_class_tp_confidence", {})
                if per_tp_conf:
                    tp_cls_parts = [f"{name}={per_tp_conf[name]:.3f}" for name in per_tp_conf]
                    print(f"  Per-class TP confidence: {', '.join(tp_cls_parts)}")
                per_fp_conf = details.get("per_class_fp_confidence", {})
                if per_fp_conf:
                    fp_cls_parts = [f"{name}={per_fp_conf[name]:.3f}" for name in per_fp_conf]
                    print(f"  Per-class FP confidence: {', '.join(fp_cls_parts)}")
                # 整體 TP/FP 平均信心
                tp_val, fp_val = details.get("tp_confidence", float("nan")), details.get("fp_confidence", float("nan"))
                tp_str = f"{tp_val:.3f}" if tp_val == tp_val else "N/A"
                fp_str = f"{fp_val:.3f}" if fp_val == fp_val else "N/A"
                print(f"  TP Confidence (correct): {tp_str}, FP Confidence (wrong): {fp_str}")

    if args.save_model_path:
        save_root = args.save_model_path
        domain = _infer_domain_for_target()
        save_dir = os.path.join(save_root, domain)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{domain}_{'_'.join(class_names)}.pth"
        save_path = os.path.join(save_dir, filename)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "extracted_layer": extracted_layer,
                "num_classes": num_classes,
                "model_class": args.model_class,
            },
            save_path,
        )
        print(f"[INFO] Saved target baseline model to: {save_path}")


if __name__ == "__main__":
    main()

