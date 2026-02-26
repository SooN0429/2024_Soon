"""
簡單檢驗特徵可分性的小腳本。

功能：
  1. 從 feature_root 底下讀取各子資料夾的 features_*.npy
     - 例如：
         /.../Source_train_badnets_clean/
           badnets/features_resnet18_7_point_K100.npy
           clean/features_resnet18_7_point_K100.npy
  2. 合併成一個 (N, D) 的特徵矩陣與 (N,) 的標籤
  3. 使用 sklearn 的 LogisticRegression 與 LinearSVC 做 train/test split
  4. 印出兩個模型在 test set 上的 accuracy，幫助判斷
     「這組特徵在理論上能不能分開 badnets / clean」

使用範例：

    python check_feature_separability.py \
      --feature_root "/media/user906/ADATA HV620S/lab/feature_poisoned_cifar-10_/source/Source_train_badnets_clean" \
      --feature_glob "features_*.npy" \
      --test_size 0.3 \
      --random_state 0
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check separability of extracted features using sklearn classifiers."
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
        "--test_size",
        type=float,
        default=0.3,
        help="train/test 切分比例中 test 的比例 (預設 0.3)。",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="隨機種子。",
    )
    return parser.parse_args()


def load_features_from_root(
    feature_root: str,
    feature_glob: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    與 source_feature_model_train.py / target_feature_baseline_train.py 的邏輯一致：
      - 每個子資料夾視為一個類別
      - 讀取第一個符合 feature_glob 的檔案
      - 合併成 features (N, D) 與 labels (N,)
    """
    class_dirs = [
        d for d in sorted(os.listdir(feature_root))
        if os.path.isdir(os.path.join(feature_root, d))
    ]
    if not class_dirs:
        raise RuntimeError(f"No subdirectories found under feature_root={feature_root}")

    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for class_idx, class_name in enumerate(class_dirs):
        subdir = os.path.join(feature_root, class_name)
        pattern = os.path.join(subdir, feature_glob)
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise RuntimeError(f"No feature files matching {feature_glob} in {subdir}")
        feat_path = candidates[0]
        feats = np.load(feat_path)

        # 支援 2D 或 4D 特徵：
        # - 2D: 直接使用 (N, D)
        # - 4D: 視為 (N, C, H, W)，flatten 成 (N, C*H*W)
        if feats.ndim == 4:
            feats = feats.reshape(feats.shape[0], -1)
        elif feats.ndim != 2:
            raise RuntimeError(
                f"Expected 2D or 4D feature array in {feat_path}, got shape {feats.shape}"
            )
        labels = np.full((feats.shape[0],), class_idx, dtype=np.int64)
        all_feats.append(feats)
        all_labels.append(labels)

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, class_dirs


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading features from: {args.feature_root}")
    X, y, class_names = load_features_from_root(args.feature_root, args.feature_glob)
    print(f"[INFO] features shape = {X.shape}, classes = {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # 標準化有助於 linear classifier 收斂
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print("[INFO] Training LogisticRegression...")
    lr_clf = LogisticRegression(max_iter=2000)
    lr_clf.fit(X_train_std, y_train)
    y_pred_lr = lr_clf.predict(X_test_std)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"[RESULT] LogisticRegression test accuracy = {acc_lr * 100:.2f}%")

    print("[INFO] Training LinearSVC...")
    svm_clf = LinearSVC(max_iter=5000)
    svm_clf.fit(X_train_std, y_train)
    y_pred_svm = svm_clf.predict(X_test_std)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"[RESULT] LinearSVC test accuracy        = {acc_svm * 100:.2f}%")


if __name__ == "__main__":
    main()

