"""
專用於 Extract_feature_map_v2.py 的特徵抽取參數設定檔（集中管理、利於實驗對照）。

用法：
- Extract_feature_map_v2.py 會 import FEATURE_EXTRACT_CFG 作為 argparse 的預設值。
- 執行時依 --domain source/target 選用 SOURCE_EXTRACT_PROFILE 或 TARGET_EXTRACT_PROFILE，
  作為 samples_per_class、extracted_layer、clean_dir 的預設值（CLI 可覆蓋）；attack_dirs / attack_dir 需由命令列二選一必填。
- 輸出的 split_name 由程式依 domain、attack_dirs、clean_dir 自動產生，格式：Domain_train_nclass(類別名)。
"""

# 通用預設（特徵抽取腳本之超參數與固定路徑）
# input_root：僅放「父目錄」，底下需有 train_source、train_target；程式依 --domain 選用其一。
# 來源/目標專用設定（樣本數等）請用 SOURCE_EXTRACT_PROFILE / TARGET_EXTRACT_PROFILE。
FEATURE_EXTRACT_CFG = {
    "input_root": "/media/user906/ADATA HV620S/lab/poisoned_Cifar-10",
    "output_root": "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_",
    "seed": 42,  # 隨機種子(挑選抽取特徵的樣本用的隨機種子)
    "batch_size": 32,
    "num_workers": 4,
    "device": "cuda",
    "backbone": "resnet18",
    "pretrained": True,
    "transform_mode": "safe_eval",
    "pooling": "none",
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
    "clean_dir": "clean", # 乾淨類別
}

TARGET_EXTRACT_PROFILE = {
    "split_name": "Target_train_3class(badnets_refool_clean)", # 輸出路徑名稱，以此名稱為例，程式抽取完的特徵會放入 output_root/domain(Source/Target)/Target_train_3class(badnets_refool_clean)
    "samples_per_class": 20, # 每個類別抽取20筆資料
    "extracted_layer": "7_point", # 抽取第7層特徵
    "clean_dir": "clean", # 乾淨類別
}
