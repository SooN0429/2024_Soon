---
name: training scripts model_class param
overview: 在 source_feature_model_train.py 與 target_feature_baseline_train.py 兩支訓練腳本中新增 --model_class 參數，並以 importlib 動態載入 O2M 的 models 或 models1，使兩腳本皆能依參數選擇不同 Transfer_Net 架構訓練，以產出供 model3class_fusion2025.py 異構融合使用的 checkpoint。
todos: []
isProject: false
---

# 訓練腳本支援可選模型架構（做法一）

## 目標

- **source_feature_model_train.py**、**target_feature_baseline_train.py** 皆支援以參數選擇 O2M 底下的 `models` 或 `models1` 建構 `Transfer_Net`，其餘訓練流程不變。
- 產出的 checkpoint 可被 [model3class_fusion2025.py](tang_Vincent_transfer/feature_extract-poisoned/model3class_fusion2025.py) 依 `--target_model_class` / `--source_model_class` 正確載入，進行異構或同構融合。

## 變更範圍

兩支腳本做**對稱**的改動，僅檔名與變數名不同。

---

### 1. [source_feature_model_train.py](tang_Vincent_transfer/feature_extract-poisoned/source_feature_model_train.py)

**1.1 頂部 import**

- 新增 `import importlib`（若尚未引入）。
- 保留 `import models` 以維持既有程式可讀性與預設行為；主流程改為依 `args.model_class` 動態取得 `Transfer_Net`。

**1.2 parse_args()**

- 在 `--device` 之後、`return parser.parse_args()` 之前，新增：
  - `--model_class`，型別 `str`，預設 `"models"`。
  - `help` 說明：可選 O2M 底下的模組名，例如 `models`（2 層 bottle）或 `models1`（4 層 bottle），用於建構 `Transfer_Net`；與 model3class_fusion2025 的 `--source_model_class` / `--target_model_class` 對應。
  - 可不限制 `choices`，僅在文件說明常用值為 `models`、`models1`；若限制則用 `choices=["models", "models1"]`。

**1.3 main() 內建構模型**

- 在設定 `backbone_multi.extracted_layer`、取得 `n_class` 之後，在 `train_loader = build_feature_dataloader(...)` 之前：
  - 使用 `importlib.import_module(args.model_class)` 取得模組（O2M 已在 `sys.path`，會載入 O2M 下的同名模組）。
  - `Transfer_Net = mod.Transfer_Net`。
  - 將原本的 `model = models.Transfer_Net(n_class)` 改為 `model = Transfer_Net(n_class)`。
  - 可加一行 `print(f"[INFO] model_class = {args.model_class}")` 方便除錯。

**1.4 儲存 checkpoint**

- 在 `torch.save(..., save_path)` 的 dict 中新增一鍵 `"model_class"`，值為 `args.model_class`，以便日後辨識該 checkpoint 的架構（與 model3class_fusion2025 的參數對應時可參考）。

---

### 2. [target_feature_baseline_train.py](tang_Vincent_transfer/feature_extract-poisoned/target_feature_baseline_train.py)

- **2.1** 頂部新增 `import importlib`。
- **2.2** 在 `parse_args()` 中新增 `--model_class`，預設 `"models"`，說明與 source 腳本一致（可註明：目標端若要用異構融合，可設為 `models1`）。
- **2.3** 在 `main()` 中，於建立 `train_loader` 之前依 `args.model_class` 動態載入模組並取得 `Transfer_Net`，將 `model = models.Transfer_Net(n_class)` 改為 `model = Transfer_Net(n_class)`，並可列印 `model_class`。
- **2.4** 在 `torch.save(...)` 的 dict 中新增 `"model_class": args.model_class`。

---

### 3. 使用方式與對應關係

- **同構融合**（兩端皆 `models`）：  
  - source / target 訓練時皆不帶 `--model_class`（或皆設 `--model_class models`），產出的兩個 checkpoint 給 model3class_fusion2025 時使用 `--source_model_class models --target_model_class models`。
- **異構融合**（target=models1, source=models）：  
  - 用 **source_feature_model_train.py** 訓練來源端時使用預設或 `--model_class models`。  
  - 用 **target_feature_baseline_train.py** 訓練目標端時加上 `--model_class models1`。  
  - 融合時使用 `--source_model_class models --target_model_class models1`（即 model3class_fusion2025 的預設）。

Checkpoint 內含 `model_class` 後，若日後在 fusion 或評估腳本中要自動推斷架構，可從 checkpoint 讀取該欄位，目前仍以命令列參數為準即可。

---

### 4. 注意事項

- 兩支腳本皆假設執行時已透過 `sys.path.insert(0, _O2M_DIR)` 載入 O2M，因此 `importlib.import_module("models")` 或 `import_module("models1")` 會載入 O2M 下的對應模組，無需改動路徑邏輯。
- 僅新增參數與動態載入，optimizer / scheduler / criterion / 訓練迴圈 / 評估 / 儲存路徑邏輯均不變。

