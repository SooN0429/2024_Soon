import glob
import os
import random
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class AttackTypeBalancedTestDataset(Dataset):
    """
    從兩層結構的測試資料夾中抽樣：

        root/
          attack_type_1/
            0/*.png
            ...
            9/*.png
          attack_type_2/
            0/*.png
            ...
            9/*.png
          ...

    對於每個 attack_type(class)、每個 digit(0-9)(CIFAR-10的類別)，最多抽 per_digit_k 張，
    並將 attack_type 映射為整數 label (0 ~ len(attack_types)-1)。
    """

    def __init__(
        self,
        root: str,
        attack_types: List[str],
        per_digit_k: int,
        transform=None,
        seed: int = 0,
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.root = root
        self.attack_types = attack_types
        self.per_digit_k = per_digit_k
        self.transform = transform
        self.exts = exts

        random.seed(seed)

        # attack_type -> label id
        self.attack_to_idx = {atk: i for i, atk in enumerate(attack_types)}

        self.samples: List[Tuple[str, int]] = []

        for atk in attack_types:
            atk_root = os.path.join(root, atk)
            if not os.path.isdir(atk_root):
                continue
            for digit in map(str, range(10)):
                digit_dir = os.path.join(atk_root, digit)
                if not os.path.isdir(digit_dir):
                    continue

                files: List[str] = []
                for ext in exts:
                    pattern = os.path.join(digit_dir, f"*{ext}")
                    files.extend(glob.glob(pattern))

                if not files:
                    continue

                random.shuffle(files)
                chosen = files if self.per_digit_k <= 0 else files[: self.per_digit_k]
                label = self.attack_to_idx[atk]

                for f in chosen:
                    self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under '{root}' for attack_types={attack_types}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

