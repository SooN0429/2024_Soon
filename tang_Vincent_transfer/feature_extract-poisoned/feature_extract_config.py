"""
專用於 Extract_feature_map_v2.py 的特徵抽取參數設定檔（集中管理、利於實驗對照）。

用法：
- Extract_feature_map_v2.py 會 import FEATURE_EXTRACT_CFG 作為 argparse 的預設值。
- SOURCE_EXTRACT_PROFILE / TARGET_EXTRACT_PROFILE 為常用實驗組合，可於 shell 或腳本中參考，
  或未來擴充為從 config 讀取 profile 再帶入 CLI。
"""

# 通用預設（特徵抽取腳本之超參數，不含路徑）
FEATURE_EXTRACT_CFG = {
    "seed": 42,
    "batch_size": 32,
    "num_workers": 4,
    "device": "cuda",
    "backbone": "resnet18",
    "pretrained": True,
    "transform_mode": "safe_eval",
    "pooling": "avg",
    "min_class_policy": "truncate",
    "max_total_samples": None,
    "save_filenames": True,
    "split_name": "train",
}

# 常用實驗 profile（僅供參考或腳本帶參用，不強制被 Extract_feature_map_v2 讀取）
SOURCE_EXTRACT_PROFILE = {
    "split_name": "Source_train_2Attack_clean",
    "samples_per_class": 100,
    "extracted_layer": "7_point",
    "attack_dirs": ["badnets", "refool"],
    "clean_dir": "clean",
}

TARGET_EXTRACT_PROFILE = {
    "split_name": "Target_train_2Attack_clean",
    "samples_per_class": 20,
    "extracted_layer": "7_point",
    "attack_dirs": ["badnets", "refool"],
    "clean_dir": "clean",
}
