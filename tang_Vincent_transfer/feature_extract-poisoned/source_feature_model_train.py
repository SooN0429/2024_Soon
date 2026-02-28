"""
用已萃取的 source 特徵 (features_*.npy) 訓練二分類模型，
並在指定的影像資料集上做測試。

訓練方式比照學姊腳本 (1_source_model_train.py)：
  - 使用 Transfer_Net（base_network + bottle_layer + classifier_layer）
  - 訓練時輸入為預萃取特徵，forward 時 test_flag=0，只跑 backbone 後段 + bottle + classifier
  - Optimizer 更新：base_network、avgpool、bottle_layer、classifier_layer（與學姊相同）
  - Loss：LabelSmoothingCrossEntropy
  - Scheduler：ExponentialLR(gamma=0.85)

本版腳本進一步自動化參數：
  - extracted_layer：若未指定，從 features_*.npy 檔名自動解析 (例如 features_resnet18_7_point_K100.npy → 7_point)
  - num_classes：由 feature_root 底下子資料夾數量自動決定
  - attack_types：由 feature_root 子資料夾名稱自動決定，確保訓練 / 測試類別一致
  - save_model_path：指定訓練模型根目錄 (例如 .../trained_model_cpt)，實際檔案會依 source/target 自動放入對應子資料夾並生成檔名

資料約定 (以你給的路徑為例)：

    --feature_root=/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/source/Source_train_badnets_clean

該目錄底下應有多個子資料夾，每個代表一個類別，例如：

    Source_train_badnets_clean/
      badnets/
        features_resnet18_7_point_K100.npy
      clean/
        features_resnet18_7_point_K100.npy

腳本會：
  1. 掃描 feature_root 下面的所有子資料夾，依字母序決定類別 id (0,1,...)
  2. 從每個子資料夾讀第一個符合 features_*.npy 的檔案，拼成一個大的特徵矩陣，並嘗試從檔名推斷 extracted_layer
  3. 用這些特徵訓練 Transfer_Net。支援兩種格式（比照學姊）：
     - 4D (N, C, H, W)：直接以 (B, C, H, W) 送入 backbone 後段（學姊方式，H,W>=7）
     - 2D (N, C)：reshape 成 (B, C, 1, 1) 送入（需 backbone 支援 1x1 或另改 avgpool）
  4. 測試時用影像經完整 backbone + bottle + classifier 評估

依賴學姊的 O2M 模組：models, config, utils, backbone_multi, call_resnet18_multi, LabelSmoothing。

#指令範例：(路徑改成自己的)
python source_feature_model_train.py \
  --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/source/Source_train_badnets_clean" \
  --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10/test" \
  --save_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt"

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
from torchvision import transforms

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

from attack_test_dataset import AttackTypeBalancedTestDataset
from eval_utils import evaluate_on_images_with_per_class
from feature_train_config import SOURCE_FEATURE_CFG as SFCFG
from feature_train_config import build_attack_balanced_test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train source model on extracted features (same setup as 學姊 script)."
    )
    parser.add_argument(  # 若未指定，會從 features_*.npy 檔名自動解析 (例如 7_point)
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
        default=SFCFG["batch_size"],
        help="訓練與測試 batch size（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=SFCFG["lr"],
        help="學習率（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=SFCFG["epoch"],
        help="訓練 epoch 數（預設來自 feature_train_config）。",
    )
    parser.add_argument(
        "--eval_image_root",
        type=str,
        required=True,
        help="用於測試的影像資料集根目錄 (例如 poisoned_Cifar-10/test)。",
    )
    parser.add_argument(
        "--per_digit_k",
        type=int,
        default=SFCFG["per_digit_k"],
        help="每個 digit(0-9) 在每個攻擊型態資料夾中抽取的最大樣本數（預設來自 feature_train_config）。",
    )
    parser.add_argument( #可優化:根據--attack_types的類別資料夾名稱 生成模型儲存路徑
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
    return parser.parse_args()


def load_features_from_root(
    feature_root: str,
    feature_glob: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
    """
    掃描 feature_root 底下的每個子資料夾，讀 features_*.npy，
    回傳 (features, labels, class_names)。
    支援 2D (N, C) 或 4D (N, C, H, W)，與學姊 1_Extract_feature_map 輸出格式一致。
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


def _infer_domain_from_feature_root(feature_root: str) -> str:
    """
    根據 feature_root 路徑判斷是 source 還是 target 端。
    規則：路徑切成小寫片段，包含 'source' 則回傳 'source'，包含 'target' 則回傳 'target'，否則預設 'source'。
    """
    norm = os.path.normpath(feature_root).lower()
    parts = norm.split(os.sep)
    if "source" in parts:
        return "source"
    if "target" in parts:
        return "target"
    return "source"


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
        # 學姊 backbone 在 test_flag=0 時從 convm2_layer 開始，輸入 (B, C, H, W)，H,W>=7
        x = x.to(device)
        y = y.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C) -> (B, C, 1, 1)
        # 若 x.dim() == 4 則保持 (B, C, H, W)，不變
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

    print(f"[INFO] Loading features from: {args.feature_root}")
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

    # 與學姊一致：設定 extracted_layer 供 backbone 使用
    backbone_multi.extracted_layer = extracted_layer

    # 分類類別數一律由 feature_root 子資料夾數決定
    num_classes = len(class_names)

    # 與學姊一致：使用 CFG['n_class']，預設為 2
    n_class = CFG.get("n_class", num_classes)
    if num_classes != n_class:
        print(
            f"[WARN] Using num_classes={num_classes} (from data), "
            f"CFG n_class={n_class} overridden for this run."
        )

    train_loader = build_feature_dataloader(features, labels, args.batch_size)

    # 與學姊一致：Transfer_Net + 同一組 optimizer / scheduler / criterion
    model = models.Transfer_Net(n_class)
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

    print(f"[INFO] Building image DataLoader from: {args.eval_image_root}")
    attack_types = class_names  # 測試時 attack_types 與訓練特徵類別一致
    image_loader = build_attack_balanced_test_loader(
        root=args.eval_image_root,
        batch_size=args.batch_size,
        attack_types=attack_types,
        per_digit_k=args.per_digit_k,
    )

    best_acc = 0.0
    log_interval = SFCFG["log_interval"]
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

    if args.save_model_path:
        # 根據 feature_root 判斷 source / target，並在 save_model_path 底下建立對應子資料夾與自動檔名
        domain = _infer_domain_from_feature_root(args.feature_root)
        save_root = args.save_model_path
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
            },
            save_path,
        )
        print(f"[INFO] Saved model to: {save_path}")


if __name__ == "__main__":
    main()
