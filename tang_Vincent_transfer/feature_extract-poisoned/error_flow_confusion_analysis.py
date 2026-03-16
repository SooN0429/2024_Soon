"""
error_flow_confusion_analysis.py

針對以下兩種設定，在共同測試集上計算 confusion matrix 與 clean 錯誤流向：
- 融合後 + 僅分類層微調（fusion_head_ckpt）
- 融合後 + 全模型微調（fusion_full_ckpt）

聚焦問題：
- clean 樣本主要被錯分成 Badnets 還是 Refool？
- 是偏向某一側，還是兩邊都吸走？

  cd /home/user906/2024_Soon/tang_Vincent_transfer/feature_extract-poisoned

  python error_flow_confusion_analysis.py \
    --fusion_head_ckpt "/media/user906/ADATA HV620S/lab/trained_model_cpt/models/target_AfterFusion/M2O_3class_para05_05_head_only_with_3classdata.pth" \
    --fusion_full_ckpt "/media/user906/ADATA HV620S/lab/trained_model_cpt/models/target_AfterFusion/M2O_3class_para05_05_full_with_3classdata.pth" \
    --eval_image_root "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10_v1/test" \
    --output_dir "/media/user906/ADATA HV620S/lab/error_flow_analysis"

"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_O2M_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "O2M"))
if _O2M_DIR not in sys.path:
    sys.path.insert(0, _O2M_DIR)

import backbone_multi  # noqa: E402
import call_resnet18_multi  # noqa: F401,E402
from config import CFG  # noqa: E402
from feature_train_config import TARGET_FEATURE_CFG as TFCFG  # noqa: E402
from feature_train_config import build_attack_balanced_test_loader  # noqa: E402


@dataclass
class ModelSpec:
    label: str
    ckpt_path: str
    model_class_key: str | None  # checkpoint 內的 model_class 欄位名稱或 None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare error flow (confusion matrix) across "
            "fusion (head/full) models."
        )
    )
    parser.add_argument(
        "--fusion_head_ckpt",
        type=str,
        required=True,
        help="融合後 + 僅分類層微調的 checkpoint (.pth)",
    )
    parser.add_argument(
        "--fusion_full_ckpt",
        type=str,
        required=True,
        help="融合後 + 全模型微調的 checkpoint (.pth)",
    )
    parser.add_argument(
        "--eval_image_root",
        type=str,
        required=True,
        help="共用測試集根目錄（包含 clean/Badnets/Refool 等子資料夾）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TFCFG["batch_size"],
        help="測試 batch size（預設來自 feature_train_config）",
    )
    parser.add_argument(
        "--per_digit_k",
        type=int,
        default=TFCFG["per_digit_k"],
        help="每個 digit 在每個攻擊資料夾中抽樣的最大樣本數",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="運算裝置，cuda 或 cpu",
    )
    parser.add_argument(
        "--clean_name",
        type=str,
        default="clean",
        help="class_names 中對應 clean 類別的名稱",
    )
    parser.add_argument(
        "--badnets_name",
        type=str,
        default="badnets",
        help="class_names 中對應 Badnets 類別的名稱",
    )
    parser.add_argument(
        "--refool_name",
        type=str,
        default="refool",
        help="class_names 中對應 Refool 類別的名稱",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./error_flow_outputs",
        help="輸出 confusion matrix 圖與統計結果的資料夾",
    )
    parser.add_argument(
        "--override_extracted_layer",
        type=str,
        default=None,
        help="若指定，會覆寫 checkpoint 內的 extracted_layer，以確保與現有模型一致",
    )
    return parser.parse_args()


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {path} has no 'state_dict' field")
    if "class_names" not in ckpt:
        raise KeyError(f"Checkpoint {path} has no 'class_names' field")
    return ckpt


def _build_union_class_names(ckpts: List[Dict]) -> List[str]:
    union_names: List[str] = []
    for ckpt in ckpts:
        names = ckpt.get("class_names", [])
        if not isinstance(names, list):
            raise ValueError("Checkpoint class_names must be a list")
        for n in names:
            if n not in union_names:
                union_names.append(n)
    if not union_names:
        raise ValueError("No class_names found across checkpoints")
    return union_names


def _ensure_three_special_classes(
    union_names: List[str],
    clean_name: str,
    badnets_name: str,
    refool_name: str,
) -> Tuple[int, int, int]:
    missing = [n for n in (clean_name, badnets_name, refool_name) if n not in union_names]
    if missing:
        raise ValueError(
            f"Required class names {missing} not all present in union_names={union_names}"
        )
    idx_clean = union_names.index(clean_name)
    idx_badnets = union_names.index(badnets_name)
    idx_refool = union_names.index(refool_name)
    return idx_clean, idx_badnets, idx_refool


def _build_test_loader(
    eval_image_root: str,
    batch_size: int,
    per_digit_k: int,
    attack_types: List[str],
) -> DataLoader:
    loader = build_attack_balanced_test_loader(
        root=eval_image_root,
        batch_size=batch_size,
        attack_types=attack_types,
        per_digit_k=per_digit_k,
    )
    return loader


def _load_transfer_net_from_ckpt(ckpt: Dict, device: torch.device) -> torch.nn.Module:
    num_classes = ckpt.get("num_classes", len(ckpt.get("class_names", [])) or CFG.get("n_class", 2))
    model_class = ckpt.get("model_class")
    if model_class is None:
        raise RuntimeError(
            "Checkpoint 缺少 'model_class' 欄位，無法自動建立 Transfer_Net。"
        )
    mod = importlib.import_module(model_class)
    Transfer_Net = mod.Transfer_Net
    model = Transfer_Net(num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


@torch.no_grad()
def evaluate_with_confusion(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    model.eval()
    test_flag = 1
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model.predict(images, test_flag)
        preds = torch.argmax(logits, dim=1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
    return cm.cpu().numpy()


def compute_clean_error_flow(
    cm: np.ndarray,
    idx_clean: int,
    idx_badnets: int,
    idx_refool: int,
) -> Dict[str, float]:
    row = cm[idx_clean]
    tp_clean = float(row[idx_clean])
    clean_to_badnets = float(row[idx_badnets])
    clean_to_refool = float(row[idx_refool])
    clean_total = float(row.sum())
    clean_to_others = float(clean_total - tp_clean - clean_to_badnets - clean_to_refool)

    denom = max(clean_to_badnets + clean_to_refool, 1.0)
    ratio_badnets = clean_to_badnets / denom
    ratio_refool = clean_to_refool / denom
    pull_index = (clean_to_badnets - clean_to_refool) / denom

    return {
        "tp_clean": tp_clean,
        "clean_total": clean_total,
        "clean_to_badnets": clean_to_badnets,
        "clean_to_refool": clean_to_refool,
        "clean_to_others": clean_to_others,
        "ratio_badnets": ratio_badnets,
        "ratio_refool": ratio_refool,
        "pull_index": pull_index,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(6, 5))
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_clean_error_bars(
    stats_per_model: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    labels = list(stats_per_model.keys())
    badnets_vals = [stats_per_model[k]["clean_to_badnets"] for k in labels]
    refool_vals = [stats_per_model[k]["clean_to_refool"] for k in labels]
    others_vals = [stats_per_model[k]["clean_to_others"] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, badnets_vals, width, label="clean→Badnets")
    plt.bar(x, refool_vals, width, label="clean→Refool")
    plt.bar(x + width, others_vals, width, label="clean→Others")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Count")
    plt.title("Clean samples error flow across models")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def print_interpretation(
    label: str,
    stats: Dict[str, float],
) -> None:
    pull = stats["pull_index"]
    cb = stats["clean_to_badnets"]
    cr = stats["clean_to_refool"]
    ct = stats["clean_total"]

    print(f"\n[INTERPRET] {label}")
    print(
        f"  clean_total={ct:.0f}, clean→Badnets={cb:.0f}, clean→Refool={cr:.0f}, "
        f"ratio_badnets={stats['ratio_badnets']:.3f}, ratio_refool={stats['ratio_refool']:.3f}, "
        f"pull_index={pull:.3f}"
    )

    if cb + cr == 0:
        print("  - 幾乎沒有 clean 被誤判為兩個攻擊類，代表 clean 邊界相對穩定。")
        return

    if abs(pull) < 0.2:
        print("  - clean 樣本同時被兩個攻擊類強烈吸引，呈現『夾在兩個私有類之間』的薄弱區。")
    elif pull > 0:
        print("  - clean 主要被 Badnets 吸走，代表 decision boundary 主要被 Badnets 一側擠壓。")
    else:
        print("  - clean 主要被 Refool 吸走，代表 decision boundary 主要被 Refool 一側擠壓。")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"[INFO] device={device}")

    # 1. 載入融合後的兩個 checkpoint
    ckpt_fusion_head = _load_checkpoint(args.fusion_head_ckpt, device)
    ckpt_fusion_full = _load_checkpoint(args.fusion_full_ckpt, device)

    ckpts = [ckpt_fusion_head, ckpt_fusion_full]

    # 2. 類別 union 與三個關鍵類別 index
    union_names = _build_union_class_names(ckpts)
    print(f"[INFO] union class_names = {union_names}")
    idx_clean, idx_badnets, idx_refool = _ensure_three_special_classes(
        union_names,
        args.clean_name,
        args.badnets_name,
        args.refool_name,
    )
    num_classes_union = len(union_names)

    # 3. extracted_layer 設定（若有的話）
    extracted_layer = args.override_extracted_layer or ckpt_fusion_full.get("extracted_layer") or "7_point"
    backbone_multi.extracted_layer = extracted_layer
    print(f"[INFO] Using extracted_layer={extracted_layer}")

    # 4. 共用 Test DataLoader（所有模型都用同一批樣本）
    image_loader = _build_test_loader(
        eval_image_root=args.eval_image_root,
        batch_size=args.batch_size,
        per_digit_k=args.per_digit_k,
        attack_types=union_names,
    )

    # 5. 為每個 checkpoint 構建模型並計算 confusion matrix
    model_specs = {
        "fusion_head": ("fusion + head_only", ckpt_fusion_head),
        "fusion_full": ("fusion + full", ckpt_fusion_full),
    }

    cms: Dict[str, np.ndarray] = {}
    stats_per_model: Dict[str, Dict[str, float]] = {}

    for key, (label, ckpt) in model_specs.items():
        print(f"\n[INFO] Building model for {label}")
        model = _load_transfer_net_from_ckpt(ckpt, device)
        cm = evaluate_with_confusion(
            model=model,
            loader=image_loader,
            device=device,
            num_classes=num_classes_union,
        )
        cms[key] = cm
        stats = compute_clean_error_flow(
            cm=cm,
            idx_clean=idx_clean,
            idx_badnets=idx_badnets,
            idx_refool=idx_refool,
        )
        stats_per_model[label] = stats

        # 個別 confusion matrix 圖
        cm_path = os.path.join(args.output_dir, f"{key}_cm.png")
        plot_confusion_matrix(
            cm=cm,
            class_names=union_names,
            title=f"{label} confusion matrix",
            save_path=cm_path,
        )

    # 6. clean 錯誤流向比較與解讀
    print("\n========== Clean error flow summary ==========")
    header = (
        f"{'Model':25s} | {'clean_total':>10s} | {'clean→Badnets':>13s} | "
        f"{'clean→Refool':>13s} | {'ratio_B':>7s} | {'ratio_R':>7s} | {'pull_idx':>8s}"
    )
    print(header)
    print("-" * len(header))
    for label in [
        "fusion + head_only",
        "fusion + full",
    ]:
        s = stats_per_model[label]
        print(
            f"{label:25s} | "
            f"{s['clean_total']:10.0f} | "
            f"{s['clean_to_badnets']:13.0f} | "
            f"{s['clean_to_refool']:13.0f} | "
            f"{s['ratio_badnets']:7.3f} | "
            f"{s['ratio_refool']:7.3f} | "
            f"{s['pull_index']:8.3f}"
        )

    # 解讀輸出：邊界擠壓 vs. 夾在兩側
    for label in [
        "fusion + head_only",
        "fusion + full",
    ]:
        print_interpretation(label, stats_per_model[label])

    # 7. clean 錯誤流向 bar chart
    bar_path = os.path.join(args.output_dir, "clean_error_flow_bar.png")
    plot_clean_error_bars(stats_per_model, bar_path)
    print(f"\n[INFO] Saved confusion matrices and clean error-flow figures to: {args.output_dir}")


if __name__ == "__main__":
    main()

