"""
model3class_fusion2025.py

流程與 model3class.py 一致：載入 source/target checkpoint → union 類別 → 載入 target 特徵 →
模型融合（2025 風格）→ 目標端特徵微調 → 影像評估 → 儲存。
融合方式參考 model_fusion_2025.py：頭/尾層 align_heterogeneous_layers，conv_M2 statistical_alignment_fusion。
使用 O2M 的 models / models1 與 backbone_multi，預設 target=models1、source=models 以支援異構融合。

指令範例：
python model3class_fusion2025.py \
  --source_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/models/source/source_badnets_clean.pth" \
  --target_model_path "/media/user906/ADATA HV620S/lab/trained_model_cpt/model1/target/target_clean_refool.pth" \
  --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/target/Target_train_2class(refool_clean)" \
  --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10_v1/test" \
  --fusion_layers all \
  --fusion_alpha 0.5 \
  --fusion_beta 0.5 \
  --statistical_method repair \
  --save_model_path "/media/user906/ADATA HV620S/lab/fusion_2025_trained_cpt/sameclass_fusion2025.pth" \
  --finetune_mode full \
  --epoch 25 \
  --seed 1

"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_O2M_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "O2M"))
if _O2M_DIR not in sys.path:
    sys.path.insert(0, _O2M_DIR)

import backbone_multi
import call_resnet18_multi  # noqa: F401
from config import CFG
import LabelSmoothing as LS

from feature_train_config import TARGET_FEATURE_CFG as TFCFG
from feature_train_config import build_attack_balanced_test_loader

# 融合演算法（與 model 類別無關）
from fusion_utils import (
    align_heterogeneous_layers,
    choose_statistical_method,
    statistical_alignment_fusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source+Target 2025-style fusion (head/tail + conv_M2), then fine-tune on target features."
    )
    parser.add_argument("--source_model_path", type=str, required=True, help="來源端 checkpoint (.pth)")
    parser.add_argument("--target_model_path", type=str, required=True, help="目標端 checkpoint (.pth)")
    parser.add_argument("--feature_root", type=str, required=True, help="target 特徵根目錄（子資料夾 = 類別）")
    parser.add_argument("--feature_glob", type=str, default="features_*.npy", help="特徵檔 glob")
    parser.add_argument("--eval_image_root", type=str, required=True, help="target 影像測試集根目錄")
    parser.add_argument("--save_model_path", type=str, default=None, help="微調後模型儲存路徑")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=TFCFG["batch_size"])
    parser.add_argument("--lr", type=float, default=TFCFG["lr"])
    parser.add_argument("--epoch", type=int, default=TFCFG["epoch"])
    parser.add_argument("--per_digit_k", type=int, default=TFCFG["per_digit_k"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--finetune_mode", type=str, default="full", choices=["full", "head_only"])

    # 模型架構（支援異構）：若未指定，將優先從 checkpoint 的 model_class 自動推斷
    parser.add_argument(
        "--target_model_class",
        type=str,
        default=None,
        help="目標端 Transfer_Net 所在模組名（O2M 下）。若不指定，將優先使用 target checkpoint 內的 model_class，否則回退為 'models1'。",
    )
    parser.add_argument(
        "--source_model_class",
        type=str,
        default=None,
        help="來源端 Transfer_Net 所在模組名（O2M 下）。若不指定，將優先使用 source checkpoint 內的 model_class，否則回退為 'models'。",
    )

    # 融合參數（對應 model_fusion_2025）
    parser.add_argument("--fusion_alpha", type=float, default=0.5, help="同構層融合權重")
    parser.add_argument("--fusion_beta", type=float, default=0.5, help="異構層融合權重")
    parser.add_argument(
        "--fusion_layers",
        type=str,
        default="all",
        choices=["all", "head_only", "tail_only", "homogeneous_only", "head_and_tail_only"],
    )
    parser.add_argument(
        "--statistical_method",
        type=str,
        default="repair",
        choices=["repair", "rescale", "original"],
    )
    parser.add_argument("--adaptive_method", type=str, default="False", help="是否依層類型自動選統計方法")
    parser.add_argument("--channel_similarity", type=str, default="True", help="是否做通道相似性對齊")
    parser.add_argument("--similarity_top_k", type=float, default=0.3)

    args = parser.parse_args()
    args.adaptive_method = args.adaptive_method.lower() in ("true", "1", "yes")
    args.channel_similarity = args.channel_similarity.lower() in ("true", "1", "yes")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features_from_root(
    feature_root: str,
    feature_glob: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
    """掃描 feature_root 子資料夾，讀 features_*.npy，回傳 (features, labels, class_names, detected_layer)。"""
    class_dirs = [
        d for d in sorted(os.listdir(feature_root))
        if os.path.isdir(os.path.join(feature_root, d))
    ]
    if not class_dirs:
        raise RuntimeError(f"No subdirectories under feature_root={feature_root}")

    all_feats, all_labels = [], []
    detected_layer = None
    for class_idx, class_name in enumerate(class_dirs):
        subdir = os.path.join(feature_root, class_name)
        pattern = os.path.join(subdir, feature_glob)
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise RuntimeError(f"No files matching {feature_glob} in {subdir}")
        feats = np.load(candidates[0])
        if feats.ndim not in (2, 4):
            raise RuntimeError(f"Expected 2D or 4D features, got ndim={feats.ndim}")
        labels = np.full((feats.shape[0],), class_idx, dtype=np.int64)
        all_feats.append(feats)
        all_labels.append(labels)
        if detected_layer is None:
            m = re.search(r"(\d+_point)", os.path.basename(candidates[0]))
            if m:
                detected_layer = m.group(1)
    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, class_dirs, detected_layer


def build_feature_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    dataset = Data.TensorDataset(x, y)
    return Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def train_one_epoch(model, loader: DataLoader, criterion, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0
    test_flag = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        y = torch.squeeze(y)
        optimizer.zero_grad()
        _, logits = model(x, y, test_flag)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_on_images_detailed(model, loader: DataLoader, device: torch.device, class_names: List[str]):
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
        images, labels = images.to(device), labels.to(device)
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
def evaluate_baseline_checkpoint(
    ckpt,
    TransferNetCls,
    class_names: List[str],
    args,
    device: torch.device,
    label: str,
) -> None:
    """
    在 target 影像測試集上評估原始 2 類 / 多類 checkpoint 的 baseline 表現，
    並列印各類別準確率與 TP/FN 平均信心（FN 為真實為該類但預測錯的信心）。
    """
    print(f"[INFO] 評估 {label} baseline（classes={class_names}）於 target 測試集上")

    image_loader = build_attack_balanced_test_loader(
        root=args.eval_image_root,
        batch_size=args.batch_size,
        attack_types=class_names,
        per_digit_k=args.per_digit_k,
    )

    num_classes = ckpt.get("num_classes", len(ckpt.get("class_names", [])) or len(class_names) or 2)
    model = TransferNetCls(num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    acc, per_class_acc, per_class_tp_conf, per_class_fn_conf = evaluate_on_images_detailed(
        model, image_loader, device, class_names
    )

    print(f"[BASELINE] {label} 在 target 測試集上的整體準確率 = {acc*100:.2f}%")
    for name in class_names:
        acc_c = per_class_acc.get(name, 0.0)
        tp = per_class_tp_conf.get(name)
        fn = per_class_fn_conf.get(name)
        tp_s = f"{tp:.4f}" if tp is not None else "N/A"
        fn_s = f"{fn:.4f}" if fn is not None else "N/A"
        print(f"  [{name}] acc={acc_c:.2f}%, TP_conf={tp_s}, FN_conf={fn_s}")


def run_fusion(
    model_0,
    model_1,
    args,
) -> None:
    """將 model_1 融合進 model_0（寫入 model_0 的 state_dict）。"""
    model = model_0

    # 頭層 / 尾層
    bottle_0 = [i for i, m in enumerate(model_0.bottle_layer) if isinstance(m, (nn.Linear, nn.Conv2d))]
    bottle_1 = [i for i, m in enumerate(model_1.bottle_layer) if isinstance(m, (nn.Linear, nn.Conv2d))]
    first_0 = bottle_0[0] if bottle_0 else None
    first_1 = bottle_1[0] if bottle_1 else None
    last_0 = bottle_0[-1] if bottle_0 else None
    last_1 = bottle_1[-1] if bottle_1 else None

    if args.fusion_layers in ("all", "head_only", "head_and_tail_only") and first_0 is not None and first_1 is not None:
        try:
            fw, fb = align_heterogeneous_layers(
                model_0.bottle_layer[first_0],
                model_1.bottle_layer[first_1],
                beta=args.fusion_beta,
            )
            model.state_dict()[f"bottle_layer.{first_0}.weight"].data.copy_(fw)
            if fb is not None:
                model.state_dict()[f"bottle_layer.{first_0}.bias"].data.copy_(fb)
            print("頭層異構融合完成")
        except Exception as e:
            print(f"頭層異構融合失敗: {e}")

    if args.fusion_layers in ("all", "tail_only", "head_and_tail_only") and last_0 is not None and last_1 is not None:
        try:
            fw, fb = align_heterogeneous_layers(
                model_0.bottle_layer[last_0],
                model_1.bottle_layer[last_1],
                beta=args.fusion_beta,
            )
            model.state_dict()[f"bottle_layer.{last_0}.weight"].data.copy_(fw)
            if fb is not None:
                model.state_dict()[f"bottle_layer.{last_0}.bias"].data.copy_(fb)
            print("尾層異構融合完成")
        except Exception as e:
            print(f"尾層異構融合失敗: {e}")

    # conv_M2 層
    for name, params_0 in model_0.state_dict().items():
        if "bottle_layer" in name or "num_batches_tracked" in name:
            continue
        if name not in model_1.state_dict():
            continue
        params_0 = params_0.data
        params_1 = model_1.state_dict()[name].data
        is_homogeneous = params_0.shape == params_1.shape

        if not is_homogeneous and args.fusion_layers == "homogeneous_only":
            continue
        if args.fusion_layers == "head_and_tail_only":
            continue

        is_conv_m2 = (
            "base_network.convm2_layer" in name
            and ("weight" in name or "bias" in name)
            and "num_batches_tracked" not in name
            and "running_mean" not in name
            and "running_var" not in name
        )

        if is_homogeneous and is_conv_m2:
            try:
                method = (
                    choose_statistical_method(name, params_0.shape, args.statistical_method)
                    if args.adaptive_method
                    else args.statistical_method
                )
                fused = statistical_alignment_fusion(
                    params_0,
                    params_1,
                    alpha=args.fusion_alpha,
                    eps=1e-5,
                    repair_type=method,
                    layer_name=name,
                    enable_channel_similarity=args.channel_similarity,
                    similarity_top_k=args.similarity_top_k,
                )
                model.state_dict()[name].data.copy_(fused)
            except Exception as e:
                print(f"conv_M2 融合失敗 {name}: {e}")
        elif not is_homogeneous and args.fusion_layers != "homogeneous_only" and is_conv_m2:
            try:
                min_dims = [min(d0, d1) for d0, d1 in zip(params_0.shape, params_1.shape)]
                slice_obj = tuple(slice(0, d) for d in min_dims)
                fused = params_0.clone()
                p0_slice = params_0[slice_obj]
                p1_slice = params_1[slice_obj]
                method = (
                    choose_statistical_method(name, p0_slice.shape, args.statistical_method)
                    if args.adaptive_method
                    else args.statistical_method
                )
                fused_slice = statistical_alignment_fusion(
                    p0_slice, p1_slice,
                    alpha=args.fusion_alpha,
                    eps=1e-5,
                    repair_type=method,
                    layer_name=name,
                    enable_channel_similarity=args.channel_similarity,
                    similarity_top_k=args.similarity_top_k,
                )
                fused[slice_obj] = fused_slice
                model.state_dict()[name].data.copy_(fused)
            except Exception as e:
                print(f"異構 conv_M2 融合失敗 {name}: {e}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] device={device}")

    # 1. 載入 checkpoints（也用來自動推斷 model_class）
    print(f"[INFO] Loading source: {args.source_model_path}")
    ckpt_src = torch.load(args.source_model_path, map_location=device)
    num_classes_src = ckpt_src.get("num_classes", len(ckpt_src.get("class_names", [])) or 2)
    extracted_layer_src = ckpt_src.get("extracted_layer")

    print(f"[INFO] Loading target: {args.target_model_path}")
    ckpt_tgt = torch.load(args.target_model_path, map_location=device)
    num_classes_tgt = ckpt_tgt.get("num_classes", len(ckpt_tgt.get("class_names", [])) or 2)
    extracted_layer_tgt = ckpt_tgt.get("extracted_layer")

    # 1.1 從 checkpoint 自動推斷 model_class，除非使用者明確指定
    src_model_class_ckpt = ckpt_src.get("model_class")
    tgt_model_class_ckpt = ckpt_tgt.get("model_class")

    if args.source_model_class is None:
        if src_model_class_ckpt is not None:
            args.source_model_class = src_model_class_ckpt
            print(f"[INFO] Auto-detected source_model_class from checkpoint: {args.source_model_class}")
        else:
            raise RuntimeError(
                "來源端 checkpoint 未包含 'model_class' 欄位。\n"
                "請確認該檔案是由 source_feature_model_train.py 產生，"
                "或重新以正確的 model_class 訓練來源模型，"
                "也可以在指令中手動指定 --source_model_class。"
            )

    if args.target_model_class is None:
        if tgt_model_class_ckpt is not None:
            args.target_model_class = tgt_model_class_ckpt
            print(f"[INFO] Auto-detected target_model_class from checkpoint: {args.target_model_class}")
        else:
            raise RuntimeError(
                "目標端 checkpoint 未包含 'model_class' 欄位。\n"
                "請確認該檔案是由 target_feature_baseline_train.py 產生，"
                "或重新以正確的 model_class 訓練目標端 baseline，"
                "也可以在指令中手動指定 --target_model_class。"
            )

    print(
        f"[INFO] target_model_class={args.target_model_class}, "
        f"source_model_class={args.source_model_class}"
    )
    print(f"[INFO] fusion_layers={args.fusion_layers}, finetune_mode={args.finetune_mode}")

    # 1.2 動態載入 O2M / CL_MAL 的 models / models1（此時已決定最終使用的 model_class）
    mod_target = importlib.import_module(args.target_model_class)
    mod_source = importlib.import_module(args.source_model_class)
    TransferNetTarget = mod_target.Transfer_Net
    TransferNetSource = mod_source.Transfer_Net

    n_class_fusion = max(num_classes_src, num_classes_tgt)
    if num_classes_src != num_classes_tgt:
        print(f"[WARN] num_classes source={num_classes_src} target={num_classes_tgt}, using n_class_fusion={n_class_fusion}")

    # 2. Union 類別
    src_names = ckpt_src.get("class_names", [])
    tgt_names = ckpt_tgt.get("class_names", [])
    if not isinstance(src_names, list) or not isinstance(tgt_names, list):
        raise ValueError("Checkpoints must contain 'class_names' list.")
    union_names = []
    for n in src_names + tgt_names:
        if n not in union_names:
            union_names.append(n)
    class_name_to_union_idx = {n: i for i, n in enumerate(union_names)}
    num_classes_union = len(union_names)
    print(f"[INFO] union class_names = {union_names}")

    # 3. 載入 target 特徵並 remap 到 union index
    print(f"[INFO] Loading features from {args.feature_root}")
    features, labels, class_names, detected_layer = load_features_from_root(args.feature_root, args.feature_glob)
    missing = [n for n in class_names if n not in class_name_to_union_idx]
    if missing:
        raise ValueError(f"Feature classes {missing} not in union {union_names}")
    remap = {i: class_name_to_union_idx[n] for i, n in enumerate(class_names)}
    labels_remapped = labels.copy()
    for old_idx, new_idx in remap.items():
        labels_remapped[labels == old_idx] = new_idx
    labels = labels_remapped

    # 4. extracted_layer（比照 model3class：在建立 / 評估任何模型前先設定 backbone_multi.extracted_layer）
    extracted_layer = extracted_layer_tgt or extracted_layer_src or detected_layer or "7_point"
    if not extracted_layer_tgt and not extracted_layer_src and detected_layer:
        print(f"[INFO] Using extracted_layer from feature filename: {detected_layer}")
    if extracted_layer is None:
        extracted_layer = "7_point"
        print(f"[WARN] Using default extracted_layer: {extracted_layer}")
    backbone_multi.extracted_layer = extracted_layer

    # 4.1 在融合與微調前，先評估來源端 / 目標端的 baseline 表現（若 class_names 可用）
    if isinstance(src_names, list) and len(src_names) >= 2:
        try:
            evaluate_baseline_checkpoint(
                ckpt=ckpt_src,
                TransferNetCls=TransferNetSource,
                class_names=src_names,
                args=args,
                device=device,
                label="來源端模型（source checkpoint）",
            )
        except Exception as e:
            print(f"[WARN] 無法計算來源端 baseline：{e}")
    else:
        print("[WARN] 來源端 checkpoint 缺少 class_names 或類別數過少，略過 baseline 評估。")

    if isinstance(tgt_names, list) and len(tgt_names) >= 2:
        try:
            evaluate_baseline_checkpoint(
                ckpt=ckpt_tgt,
                TransferNetCls=TransferNetTarget,
                class_names=tgt_names,
                args=args,
                device=device,
                label="目標端模型（target checkpoint）",
            )
        except Exception as e:
            print(f"[WARN] 無法計算目標端 baseline：{e}")
    else:
        print("[WARN] 目標端 checkpoint 缺少 class_names 或類別數過少，略過 baseline 評估。")

    # 5. 建立 model_0（target）、model_1（source），載入權重
    model_0 = TransferNetTarget(n_class_fusion).to(device)
    model_1 = TransferNetSource(n_class_fusion).to(device)
    model_0.load_state_dict(ckpt_tgt["state_dict"], strict=False)
    model_1.load_state_dict(ckpt_src["state_dict"], strict=False)

    # 6. 融合（寫入 model_0）
    print("\n[INFO] Running 2025-style fusion (head/tail + conv_M2)...")
    run_fusion(model_0, model_1, args)

    # 7. 最終模型：target 架構、num_classes_union，複製 backbone+bottle，classifier 隨機
    model = TransferNetTarget(num_classes_union).to(device)
    dst = model.state_dict()
    src = model_0.state_dict()
    for k in dst:
        if k in src and dst[k].shape == src[k].shape:
            dst[k] = src[k].clone()
    model.load_state_dict(dst, strict=False)
    print(f"[INFO] Built final model: {args.target_model_class}.Transfer_Net({num_classes_union})")

    # 8. DataLoader / Optimizer / Criterion
    train_loader = build_feature_dataloader(features, labels, args.batch_size)
    criterion = LS.LabelSmoothingCrossEntropy(reduction="sum")

    if args.finetune_mode == "head_only":
        for p in model.base_network.parameters():
            p.requires_grad = False
        for p in model.base_network.avgpool.parameters():
            p.requires_grad = False
        for p in model.bottle_layer.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(
            [{"params": model.classifier_layer.parameters(), "lr": 10 * args.lr}],
            lr=args.lr,
            betas=CFG["betas"],
            weight_decay=CFG["l2_decay"],
        )
    else:
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85, verbose=False)

    image_loader = build_attack_balanced_test_loader(
        root=args.eval_image_root,
        batch_size=args.batch_size,
        attack_types=union_names,
        per_digit_k=args.per_digit_k,
    )

    # 9. 訓練 + 評估
    log_interval = TFCFG.get("log_interval", 10)
    best_acc = 0.0
    for epoch in range(1, args.epoch + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        acc, per_class_acc, per_class_tp, per_class_fn = evaluate_on_images_detailed(
            model, image_loader, device, union_names
        )
        if acc > best_acc:
            best_acc = acc
        if (epoch - 1) % log_interval == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}/{args.epoch:03d}] train_loss={train_loss:.6f}, eval_acc={acc*100:.2f}% (best={best_acc*100:.2f}%)")
            for name in union_names:
                tp_s = f"{per_class_tp[name]:.4f}" if per_class_tp[name] is not None else "N/A"
                fn_s = f"{per_class_fn[name]:.4f}" if per_class_fn[name] is not None else "N/A"
                print(f"  [{name}] acc={per_class_acc[name]:.2f}%, TP_conf={tp_s}, FN_conf={fn_s}")

    # 10. 儲存
    if args.save_model_path:
        os.makedirs(os.path.dirname(args.save_model_path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": union_names,
                "extracted_layer": extracted_layer,
                "num_classes": num_classes_union,
                "target_model_class": args.target_model_class,
                "source_model_class": args.source_model_class,
                "fusion_alpha": args.fusion_alpha,
                "fusion_beta": args.fusion_beta,
                "fusion_layers": args.fusion_layers,
                "finetune_mode": args.finetune_mode,
            },
            args.save_model_path,
        )
        print(f"[INFO] Saved to {args.save_model_path}")


if __name__ == "__main__":
    main()
