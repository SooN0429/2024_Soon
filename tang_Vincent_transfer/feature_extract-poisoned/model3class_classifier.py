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

新增參數功能：
--classifier_init_mode random \ # 分類層的初始化方式
--fusion_backbone_mode full_backbone \ # 決定模型中哪個部分使用學長的通道對齊與統計對齊

python model3class_classifier.py \
  --source_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/models/source/source_badnets_clean.pth" \
  --target_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/models/target/target_clean_refool.pth" \
  --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/target/Target_train_3class(badnets_refool_clean)" \
  --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10_v1/test" \
  --para_source 0.5 \
  --para_target 0.5 \
  --save_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/models/target_AfterFusion/M2O_3class_para05_05_bottle_cls_with_3classdata.pth" \
  --finetune_mode bottle_cls \
  --classifier_init_mode from_ckpt \
  --fusion_backbone_mode convM2_only \
  --seed 1
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import random
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

from fusion_utils import (
    choose_statistical_method,
    statistical_alignment_fusion,
)


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
        "--seed",
        type=int,
        default=42,
        help="隨機種子（控制初始化與 DataLoader shuffle）。",
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
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="full",
        choices=["full", "head_only", "bottle_cls"],
        help=(
            "微調模式：'full' = 整個模型微調（預設）；"
        "'head_only' = 僅微調 3 類分類頭（backbone、bottle 凍結）；"
        "'bottle_cls' = 僅微調 bottle layer + 3 類分類頭（backbone 凍結）。"
        ),
    )
    parser.add_argument(
        "--fusion_backbone_mode",
        type=str,
        default="convM2_only",
        choices=["convM2_only", "full_backbone"],
        help=(
            "在 fusion 模式下，backbone 使用 2025-style statistical_alignment_fusion 的範圍："
            "'convM2_only' = 僅對 base_network.convm2_layer.* 使用統計+通道對齊，其餘層線性加權；"
            "'full_backbone' = 對整個 base_network.* 使用統計+通道對齊，bottle_layer / classifier_layer 維持線性加權。"
        ),
    )
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="2025-style 統計對齊的融合權重 alpha（僅在 fusion 模式有效）。",
    )
    parser.add_argument(
        "--statistical_method",
        type=str,
        default="repair",
        choices=["repair", "rescale", "original"],
        help="statistical_alignment_fusion 的 repair_type。",
    )
    parser.add_argument(
        "--adaptive_method",
        type=str,
        default="False",
        help="若為 True，則每層透過 choose_statistical_method 自動決定 repair_type。",
    )
    parser.add_argument(
        "--channel_similarity",
        type=str,
        default="True",
        help="是否在 statistical_alignment_fusion 中啟用通道相似性對齊。",
    )
    parser.add_argument(
        "--similarity_top_k",
        type=float,
        default=0.3,
        help="通道相似度對齊時選取的 top-k 比例（例如 0.3 代表前 30%）。",
    )
    parser.add_argument(
        "--classifier_init_mode",
        type=str,
        default="from_ckpt",
        choices=["random", "from_ckpt"],
        help=(
            "3 類（或多類）分類頭的初始化方式："
            "'random' = 保留隨機初始化；"
            "'from_ckpt' = 依 source/target checkpoint 的 classifier_layer "
            "及 class_names 進行列對齊與（必要時）取平均。"
        ),
    )
    args = parser.parse_args()
    args.adaptive_method = args.adaptive_method.lower() in ("true", "1", "yes")
    args.channel_similarity = args.channel_similarity.lower() in ("true", "1", "yes")
    return args


def set_seed(seed: int) -> None:
    """
    設定 Python / NumPy / PyTorch 的隨機種子，方便做跨次實驗穩定度比較。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
) -> Tuple[float, dict, dict, dict]:
    """
    測試時用影像，回傳整體準確率、各類別準確率、TP 信心、FN 信心。
    TP Confidence：僅預測正確的樣本，計算其 max(P) 的平均。
    FN Confidence：僅「真實為該類別但預測錯誤」的樣本，計算其 max(P) 的平均；無此類錯誤樣本時為 None。

    注意：此處的 FN 是以「真實類別」為主的錯誤統計（label-based）。
    """
    model.eval()
    test_flag = 1
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    tp_conf_sum = [0.0] * num_classes
    tp_count = [0] * num_classes
    fn_conf_sum = [0.0] * num_classes
    fn_count = [0] * num_classes

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model.predict(images, test_flag)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        max_p = probs.max(dim=1)[0]

        for i in range(labels.size(0)): 
            c = labels[i].item()
            class_total[c] += 1
            correct = preds[i].item() == c
            conf = max_p[i].item()
            if correct:
                class_correct[c] += 1
                tp_conf_sum[c] += conf
                tp_count[c] += 1
            else:
                fn_conf_sum[c] += conf
                fn_count[c] += 1

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_acc = total_correct / total_samples if total_samples else 0.0

    per_class_acc = {}
    per_class_tp_conf = {}
    per_class_fn_conf = {}
    for c in range(num_classes):
        name = class_names[c]
        n = class_total[c]
        per_class_acc[name] = (class_correct[c] / n * 100.0) if n else 0.0
        per_class_tp_conf[name] = (tp_conf_sum[c] / tp_count[c]) if tp_count[c] else None
        per_class_fn_conf[name] = (fn_conf_sum[c] / fn_count[c]) if fn_count[c] else None

    return overall_acc, per_class_acc, per_class_tp_conf, per_class_fn_conf


@torch.no_grad()
def evaluate_2class_checkpoint(
    ckpt,
    class_names_2: List[str],
    eval_image_root: str,
    batch_size: int,
    per_digit_k: int,
    device: torch.device,
) -> Tuple[float, dict, dict, dict, dict]:
    """
    給定 2 類 checkpoint 與其類別名稱，建立對應的 2 類 Transfer_Net
    與 target test DataLoader，回傳 (accuracy, per_class_acc, per_class_conf, per_class_tp_conf, per_class_fn_conf)。
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

    acc, per_class_acc, per_class_tp_conf, per_class_fn_conf = evaluate_on_images_detailed(
        model_2, loader_2, device, class_names_2
    )
    # 維持原介面：第三項為可列印的 conf（以 TP conf 代表，無則 0.0）
    per_class_conf = {
        name: (per_class_tp_conf[name] if per_class_tp_conf[name] is not None else 0.0)
        for name in class_names_2
    }
    return acc, per_class_acc, per_class_conf, per_class_tp_conf, per_class_fn_conf


def init_classifier_from_src_tgt(
    model,
    ckpt_src,
    ckpt_tgt,
    union_names: List[str],
    device: torch.device,
) -> None:
    """
    使用 source / target checkpoint 的 classifier_layer 來初始化目前 model 的 classifier_layer。

    - 若某類別只在 source 中出現 → 用 source 對應 row（weight/bias 全複製）
    - 若某類別只在 target 中出現 → 用 target 對應 row
    - 若同時出現在 source 與 target → 該 row 取兩者平均

    若遇到 union_names 中的類別在 source/target 的 class_names 都不存在，
    視為嚴重錯誤，直接拋例外並以中文提示使用者。
    """
    src_names = ckpt_src.get("class_names", [])
    tgt_names = ckpt_tgt.get("class_names", [])

    if not isinstance(src_names, list) or not isinstance(tgt_names, list):
        print("[WARN] 無法從 source/target 初始化分類層：'class_names' 欄位缺失或格式錯誤。")
        return

    state_src = ckpt_src.get("state_dict", {})
    state_tgt = ckpt_tgt.get("state_dict", {})

    w_src = state_src.get("classifier_layer.weight", None)
    b_src = state_src.get("classifier_layer.bias", None)
    w_tgt = state_tgt.get("classifier_layer.weight", None)
    b_tgt = state_tgt.get("classifier_layer.bias", None)

    if w_src is None or b_src is None or w_tgt is None or b_tgt is None:
        print("[WARN] 無法從 source/target 初始化分類層：找不到 classifier_layer 的 weight/bias。")
        return

    if w_src.shape[0] != len(src_names) or w_tgt.shape[0] != len(tgt_names):
        print(
            "[WARN] 無法從 source/target 初始化分類層："
            f"w_src.shape[0]={w_src.shape[0]}, len(src_names)={len(src_names)}, "
            f"w_tgt.shape[0]={w_tgt.shape[0]}, len(tgt_names)={len(tgt_names)}。"
        )
        return

    src_idx = {name: i for i, name in enumerate(src_names)}
    tgt_idx = {name: i for i, name in enumerate(tgt_names)}

    clf = model.classifier_layer
    W = clf.weight.data.to(device)
    b = clf.bias.data.to(device)

    num_classes_union = len(union_names)
    if W.shape[0] != num_classes_union:
        print(
            "[WARN] 無法從 source/target 初始化分類層："
            f"model.classifier_layer.weight 的列數為 {W.shape[0]}，"
            f"但 union 類別數為 {num_classes_union}。"
        )
        return

    for u_idx, name in enumerate(union_names):
        rows_w = []
        rows_b = []

        if name in src_idx:
            idx_s = src_idx[name]
            rows_w.append(w_src[idx_s].to(device))
            rows_b.append(b_src[idx_s].to(device))

        if name in tgt_idx:
            idx_t = tgt_idx[name]
            rows_w.append(w_tgt[idx_t].to(device))
            rows_b.append(b_tgt[idx_t].to(device))

        if not rows_w:
            # 嚴重錯誤：union 中出現了來源端與目標端都沒有見過的「陌生類別」
            raise RuntimeError(
                "在建立 3 類分類頭時發現陌生類別："
                f"'{name}' 不存在於 source 或 target checkpoint 的 class_names 中。\n"
                "請確認：\n"
                "1. source/target checkpoint 的 'class_names' 是否正確，且包含所有期望的類別名稱。\n"
                "2. 若你有更改 feature_root 或類別命名，請確保與 source/target 的 class_names 一致。"
            )

        if len(rows_w) == 1:
            W[u_idx].copy_(rows_w[0])
            b[u_idx].copy_(rows_b[0])
        else:
            avg_w = sum(rows_w) / len(rows_w)
            avg_b = sum(rows_b) / len(rows_b)
            W[u_idx].copy_(avg_w)
            b[u_idx].copy_(avg_b)

    clf.weight.data.copy_(W)
    clf.bias.data.copy_(b)
    print("[INFO] 已根據 source/target checkpoint 的 class_names 初始化 classifier_layer。")


def fuse_backbone_with_statistical_alignment(
    model_src: models.Transfer_Net,
    model_tgt: models.Transfer_Net,
    model_out: models.Transfer_Net,
    args,
    device: torch.device,
) -> None:
    """
    使用 2025-style statistical_alignment_fusion 對 backbone 權重做融合，並寫入 model_out。

    - 若 fusion_backbone_mode = 'convM2_only'：
        僅對 base_network.convm2_layer.* 的權重 / bias 使用 statistical_alignment_fusion，
        其他 base_network.* 參數沿用現有線性加權 (para_source / para_target)。
    - 若 fusion_backbone_mode = 'full_backbone'：
        對所有 base_network.* 權重 / bias 使用 statistical_alignment_fusion，
        其他層（bottle_layer / classifier_layer 等）維持線性加權。
    """
    state_src = model_src.state_dict()
    state_tgt = model_tgt.state_dict()
    state_out = model_out.state_dict()

    fused_state = {}
    for name, dst_param in state_out.items():
        in_src = name in state_src
        in_tgt = name in state_tgt

        # 僅考慮同名同 shape 的參數，其餘直接保持 dst_param
        if not (in_src and in_tgt):
            fused_state[name] = dst_param.clone()
            continue
        src_param = state_src[name].to(device)
        tgt_param = state_tgt[name].to(device)
        if src_param.shape != tgt_param.shape or src_param.shape != dst_param.shape:
            fused_state[name] = dst_param.clone()
            continue

        is_backbone = name.startswith("base_network.")
        is_convm2 = "base_network.convm2_layer" in name and (
            "weight" in name or "bias" in name
        )

        # 判斷是否要用 statistical_alignment_fusion
        use_stat_fusion = False
        if is_backbone:
            if args.fusion_backbone_mode == "full_backbone":
                use_stat_fusion = True
            elif args.fusion_backbone_mode == "convM2_only" and is_convm2:
                use_stat_fusion = True

        if use_stat_fusion:
            try:
                method = (
                    choose_statistical_method(name, src_param.shape, args.statistical_method)
                    if args.adaptive_method
                    else args.statistical_method
                )
                fused = statistical_alignment_fusion(
                    src_param,
                    tgt_param,
                    alpha=args.fusion_alpha,
                    eps=1e-5,
                    repair_type=method,
                    layer_name=name,
                    enable_channel_similarity=args.channel_similarity,
                    similarity_top_k=args.similarity_top_k,
                )
                fused_state[name] = fused.to(device)
            except Exception as e:
                print(f"[WARN] statistical_alignment_fusion 失敗 ({name})，改用線性加權：{e}")
                fused_state[name] = (
                    src_param * args.para_source + tgt_param * args.para_target
                )
        else:
            # 預設：線性加權融合
            fused_state[name] = src_param * args.para_source + tgt_param * args.para_target

    model_out.load_state_dict(fused_state, strict=False)


def main() -> None:
    args = parse_args()
    # 設定隨機種子，讓多次實驗結果更可重現
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] backbone_init_mode = {args.backbone_init_mode}")
    print(f"[INFO] finetune_mode = {args.finetune_mode}")

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

    # 建立來源與目標模型類別名稱的 union，作為最終 head 的類別空間
    src_names = ckpt_src.get("class_names", [])
    tgt_names = ckpt_tgt.get("class_names", [])
    if not isinstance(src_names, list) or not isinstance(tgt_names, list):
        raise ValueError("Both source and target checkpoints must contain 'class_names' as a list.")

    union_names: List[str] = []
    for name in src_names:
        if name not in union_names:
            union_names.append(name)
    for name in tgt_names:
        if name not in union_names:
            union_names.append(name)
    class_name_to_union_idx = {name: idx for idx, name in enumerate(union_names)}
    num_classes_union = len(union_names)

    print(f"[INFO] union class_names (source ∪ target) = {union_names}")

    # 2. 載入 target 特徵
    print(f"[INFO] Loading target features from: {args.feature_root}")
    features, labels, class_names, detected_layer = load_features_from_root(
        args.feature_root, args.feature_glob
    )
    print(f"[INFO] Target features shape = {features.shape}, classes = {class_names}")

    # 將 feature_root 子資料夾映射到 union 類別空間的 index
    missing_in_union = [name for name in class_names if name not in class_name_to_union_idx] 
    if missing_in_union:
        raise ValueError(
            f"The following feature classes are not present in union class_names derived from "
            f"source/target checkpoints: {missing_in_union}. "
            f"Please ensure feature_root subdirectories match source/target class_names."
        )
    remap = {old_idx: class_name_to_union_idx[name] for old_idx, name in enumerate(class_names)}
    labels_remapped = labels.copy()
    for old_idx, new_idx in remap.items():
        labels_remapped[labels == old_idx] = new_idx
    labels = labels_remapped

    num_feature_classes = len(class_names)
    n_class_cfg = CFG.get("n_class", num_feature_classes)
    if num_feature_classes != n_class_cfg:
        print(
            f"[WARN] Using num_classes={num_feature_classes} (from data), "
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
                acc_src_2, src_per_class_acc, src_per_class_conf, src_tp_conf, src_fn_conf = evaluate_2class_checkpoint(
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
                tp_parts = [f"{name}={src_tp_conf[name]:.4f}" if src_tp_conf[name] is not None else f"{name}=N/A" for name in src_classes_2]
                fn_parts = [f"{name}={src_fn_conf[name]:.4f}" if src_fn_conf[name] is not None else f"{name}=N/A" for name in src_classes_2]
                print(f"  正確預測信心 (TP Confidence): {', '.join(tp_parts)}")
                print(f"  真實為該類但預測錯 (FN Confidence): {', '.join(fn_parts)}")
            except Exception as e:
                print(f"[WARN] Failed to compute source 2-class baseline: {e}")

            try:
                acc_tgt_2, tgt_per_class_acc, tgt_per_class_conf, tgt_tp_conf, tgt_fn_conf = evaluate_2class_checkpoint(
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
                tp_parts = [f"{name}={tgt_tp_conf[name]:.4f}" if tgt_tp_conf[name] is not None else f"{name}=N/A" for name in tgt_classes_2]
                fn_parts = [f"{name}={tgt_fn_conf[name]:.4f}" if tgt_fn_conf[name] is not None else f"{name}=N/A" for name in tgt_classes_2]
                print(f"  正確預測信心 (TP Confidence): {', '.join(tp_parts)}")
                print(f"  真實為該類但預測錯 (FN Confidence): {', '.join(fn_parts)}")
            except Exception as e:
                print(f"[WARN] Failed to compute target 2-class baseline: {e}")
        else:
            print(
                "[WARN] Cannot compute 2-class baselines: "
                "ckpt_src/class_names or ckpt_tgt/class_names missing or not 2-class."
            )

        print(
            f"[INFO] Fusing Transfer_Net with para_source={args.para_source}, "
            f"para_target={args.para_target}, fusion_backbone_mode={args.fusion_backbone_mode}"
        )
        # 先建立 2 類的 source/target 模型
        model_src_2 = models.Transfer_Net(n_class_fusion).to(device)
        model_src_2.load_state_dict(ckpt_src["state_dict"], strict=False)

        model_tgt_2 = models.Transfer_Net(n_class_fusion).to(device)
        model_tgt_2.load_state_dict(ckpt_tgt["state_dict"], strict=False)

        # 建立最終 union 類別數的模型，並以 target checkpoint 作為初始值
        model = models.Transfer_Net(num_classes_union).to(device)
        state_tgt_full = ckpt_tgt["state_dict"]
        dst_state = model.state_dict()
        for name in dst_state.keys():
            if name in state_tgt_full and state_tgt_full[name].shape == dst_state[name].shape:
                dst_state[name] = state_tgt_full[name].clone()
        model.load_state_dict(dst_state, strict=False)

        # 使用 2025-style 統計對齊對 backbone 進行融合，其餘層維持線性加權
        fuse_backbone_with_statistical_alignment(
            model_src=model_src_2,
            model_tgt=model_tgt_2,
            model_out=model,
            args=args,
            device=device,
        )

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

        model = models.Transfer_Net(num_classes_union)
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
        # scratch：完全不載入任何 checkpoint，隨機初始化模型（使用 union 類別空間）
        print("[INFO] Initializing model from scratch (no pretrained checkpoint).")
        model = models.Transfer_Net(num_classes_union)
        model = model.to(device)

    # 根據 classifier_init_mode 決定是否使用 source/target 的 2 類分類頭權重
    # 依 union_names 初始化 3 類（或多類）分類頭
    if args.classifier_init_mode == "from_ckpt":
        init_classifier_from_src_tgt(
            model=model,
            ckpt_src=ckpt_src,
            ckpt_tgt=ckpt_tgt,
            union_names=union_names,
            device=device,
        )
    else:
        print("[INFO] classifier_init_mode = 'random'：保持隨機初始化的分類頭。")

    # 4. 構建 DataLoader、Optimizer、Scheduler、Loss
    train_loader = build_feature_dataloader(features, labels, args.batch_size)

    if args.finetune_mode == "head_only":
        for p in model.base_network.parameters():
            p.requires_grad = False
        for p in model.base_network.avgpool.parameters():
            p.requires_grad = False
        for p in model.bottle_layer.parameters():
            p.requires_grad = False
        print("[INFO] finetune_mode = head_only: 僅訓練 classifier_layer，backbone 與 bottle 已凍結。")
        optimizer = torch.optim.Adam(
            [{"params": model.classifier_layer.parameters(), "lr": 10 * args.lr}],
            lr=args.lr,
            betas=CFG["betas"],
            weight_decay=CFG["l2_decay"],
        )
    elif args.finetune_mode == "bottle_cls":
        for p in model.base_network.parameters():
            p.requires_grad = False
        for p in model.base_network.avgpool.parameters():
            p.requires_grad = False
        # 只開放 bottle_layer + classifier_layer
        print("[INFO] finetune_mode = bottle_cls: 僅訓練 bottle_layer + classifier_layer，backbone 已凍結。")
        optimizer = torch.optim.Adam(
            [
                {"params": model.bottle_layer.parameters(), "lr": 10 * args.lr},
                {"params": model.classifier_layer.parameters(), "lr": 10 * args.lr},
            ],
            lr=args.lr,
            betas=CFG["betas"],
            weight_decay=CFG["l2_decay"],
        )
    else:
        print("[INFO] finetune_mode = full: 整個模型微調。")
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
    attack_types = union_names
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
        acc, per_class_acc, per_class_tp_conf, per_class_fn_conf = evaluate_on_images_detailed(
            model=model,
            loader=image_loader,
            device=device,
            class_names=union_names,
        )
        if acc > best_acc:
            best_acc = acc
        if (epoch - 1) % log_interval == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}/{args.epoch:03d}] "
                f"train_loss={train_loss:.6f}, eval_acc={acc*100:.2f}% (best={best_acc*100:.2f}%)"
            )
            for name in union_names:
                tp_str = f"{per_class_tp_conf[name]:.4f}" if per_class_tp_conf[name] is not None else "N/A"
                fn_str = f"{per_class_fn_conf[name]:.4f}" if per_class_fn_conf[name] is not None else "N/A"
                print(
                    f"  [{name}] acc={per_class_acc[name]:.2f}%, "
                    f"TP_conf={tp_str}, FN_conf={fn_str}"
                )

    # 6. 儲存模型
    if args.save_model_path:
        save_dir = os.path.dirname(args.save_model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": union_names,
                "extracted_layer": extracted_layer,
                "num_classes": num_classes_union,
                "para_source": args.para_source,
                "para_target": args.para_target,
                "backbone_init_mode": args.backbone_init_mode,
                "finetune_mode": args.finetune_mode,
                # 保留最終模型所使用的架構模組名稱（目前為 models.Transfer_Net）半寫死的，這個腳本原則上是進行同構融合，所以模型架構固定為models。
                "model_class": getattr(models, "__name__", "models"),
            },
            args.save_model_path,
        )
        print(f"[INFO] Saved fused/single-backbone+fine-tuned target model to: {args.save_model_path}")


if __name__ == "__main__":
    main()