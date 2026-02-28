"""
共用驗證工具：在影像上評估模型，並回傳整體準確率與各類別辨識準確率、各類別平均信心程度。
供 source_feature_model_train、target_feature_baseline_train、
O2M_feature_transfer_train、M2O_feature_transfer_train 使用。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_on_images_with_per_class(
    model,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    test_flag: int = 1,
) -> Tuple[float, Optional[Dict]]:
    """
    在影像 DataLoader 上評估模型，回傳整體準確率與各類別準確率、各類別平均信心。

    Args:
        model: 具 predict(images, test_flag) 的模型，回傳 logits (B, num_classes)。
        loader: 回傳 (images, labels) 的 DataLoader，labels 為 0 ~ num_classes-1。
        device: 運算裝置。
        class_names: 類別名稱列表，與 label 索引對應；若為 None 或空則只回傳整體準確率，details 為 None。
        test_flag: 傳給 model.predict 的 test_flag（預設 1）。

    Returns:
        (overall_acc, details):
        - overall_acc: 整體準確率 (0~1)。
        - details: 若 class_names 有效則為 {"per_class_acc": {name: acc}, "per_class_confidence": {name: conf}}；
          若某類在測試集中無樣本則不列入該 dict。
    """
    model.eval()
    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_confidences: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model.predict(images, test_flag)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        # 每個樣本的信心 = 預測類別的機率
        confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_confidences.append(confidences.cpu())

    labels_cat = torch.cat(all_labels, dim=0)
    preds_cat = torch.cat(all_preds, dim=0)
    confidences_cat = torch.cat(all_confidences, dim=0)

    total = labels_cat.numel()
    if total == 0:
        return 0.0, None

    overall_acc = (preds_cat == labels_cat).sum().item() / total

    details = None
    if class_names and len(class_names) > 0:
        per_class_acc: Dict[str, float] = {}
        per_class_confidence: Dict[str, float] = {}
        for c in range(len(class_names)):
            mask = labels_cat == c
            n_c = mask.sum().item()
            if n_c == 0:
                continue
            acc_c = ((preds_cat == labels_cat) & mask).sum().item() / n_c
            conf_c = confidences_cat[mask].mean().item()
            per_class_acc[class_names[c]] = acc_c
            per_class_confidence[class_names[c]] = conf_c
        details = {
            "per_class_acc": per_class_acc,
            "per_class_confidence": per_class_confidence,
        }

    return overall_acc, details
