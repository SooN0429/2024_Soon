"""
專用於 Extract_feature_map_v2.py 的特徵抽取參數設定檔（集中管理、利於實驗對照）。

用法：
- Extract_feature_map_v2.py 會 import FEATURE_EXTRACT_CFG 作為 argparse 的預設值。
- SOURCE_EXTRACT_PROFILE / TARGET_EXTRACT_PROFILE 為常用實驗組合，可於 shell 或腳本中參考，
  或未來擴充為從 config 讀取 profile 再帶入 CLI。
"""

# 通用預設（特徵抽取腳本之超參數與固定路徑）
# 路徑：若不更動機器/磁碟結構，通常只改底下程式產生的資料夾名（如 split_name），可不傳 CLI 覆蓋。
FEATURE_EXTRACT_CFG = {
    "input_root": "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10/train_target",
    "output_root": "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_",
    "seed": 42,  # 隨機種子(挑選抽取特徵的樣本用的隨機種子)
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
    "split_name": "Source_train_2Attack_clean", # 輸出路徑名稱，以此名稱為例，程式抽取完的特徵會放入 output_root/domain(Source/Target)/Source_train_2Attack_clean
    "samples_per_class": 100, # 每個類別抽取100筆資料
    "extracted_layer": "7_point", # 抽取第7層特徵
    "attack_dirs": ["badnets", "refool"], # 攻擊類別
    "clean_dir": "clean", # 乾淨類別
}

TARGET_EXTRACT_PROFILE = {
    "split_name": "Target_train_3class(badnets_refool_clean)", # 輸出路徑名稱，以此名稱為例，程式抽取完的特徵會放入 output_root/domain(Source/Target)/Target_train_3class(badnets_refool_clean)
    "samples_per_class": 20, # 每個類別抽取20筆資料
    "extracted_layer": "7_point", # 抽取第7層特徵
    "attack_dirs": ["badnets", "refool"], # 攻擊類別
    "clean_dir": "clean", # 乾淨類別
}
