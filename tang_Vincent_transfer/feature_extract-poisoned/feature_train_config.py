"""
專用於 feature_extract-poisoned 目錄下的特徵訓練腳本之超參數 / 測試資料設定：

- source_feature_model_train.py
- target_feature_baseline_train.py

注意：
- 與 O2M 原本的 config.py (CFG) 分工不同：
  - config.py：仍供學姊 O2M 腳本使用 (betas, l2_decay, log_interval, n_class 等)。
  - 本檔：僅為特徵訓練腳本提供預設的 batch_size / lr / epoch / per_digit_k 等，
    以及共用的測試資料 DataLoader 組態。
"""

from torch.utils.data import DataLoader
from torchvision import transforms

from attack_test_dataset import AttackTypeBalancedTestDataset


SOURCE_FEATURE_CFG = {
    "batch_size": 16,
    "lr": 0.0001,
    "epoch": 50,
    "per_digit_k": 50,
    "log_interval": 1,
}

TARGET_FEATURE_CFG = {
    "batch_size": 16,
    "lr": 0.0001,
    "epoch": 50,
    "per_digit_k": 50,
    "log_interval": 1,
}


TEST_TRANSFORM_CFG = {
    "resize": (224, 224),
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "seed": 4,
    "num_workers": 2, 
    "drop_last": False,
}


def build_attack_balanced_test_loader(
    root: str,
    batch_size: int,
    attack_types,
    per_digit_k: int,
    cfg: dict | None = None,
) -> DataLoader:
    """
    建立供 source / target feature 訓練腳本共用的 Test DataLoader。
    """
    c = TEST_TRANSFORM_CFG if cfg is None else cfg
    transform = transforms.Compose(
        [
            transforms.Resize(c["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=c["mean"], std=c["std"]),
        ]
    )
    dataset = AttackTypeBalancedTestDataset(
        root=root,
        attack_types=attack_types,
        per_digit_k=per_digit_k,
        transform=transform,
        seed=c["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=c["num_workers"],
        drop_last=c["drop_last"],
    )
    return loader

