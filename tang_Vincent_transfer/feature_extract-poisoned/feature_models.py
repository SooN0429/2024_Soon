import torch
import torch.nn as nn
from torchvision import models


class ResNet18_7Point(nn.Module):
    """
    ResNet-18 backbone，輸出與 Extract_feature_map_v2 中
    backbone=resnet18, extracted_layer=7_point, pooling=avg 對應的 256 維特徵。

    結構：conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → avgpool → flatten
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        # 不使用 layer4，以確保輸出維度為 256
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)          # (B, 256, 1, 1)
        x = torch.flatten(x, 1)      # (B, 256)
        return x


class FeatureClassifier(nn.Module):
    """
    針對 256 維特徵的簡單分類器。
    結構大致參考舊版 Transfer_Net 的 bottle_layer + classifier_layer。
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        bottleneck_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.features_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_classes),
        )

        # 初始化，模仿舊 code 的小權重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_head(x)
        logits = self.classifier(x)
        return logits


def fuse_classifiers(
    model_a: FeatureClassifier,
    model_b: FeatureClassifier,
    weight_a: float,
    weight_b: float,
) -> FeatureClassifier:
    """
    以權重線性組合兩個 FeatureClassifier 的參數，回傳新的模型。
    new_param = weight_a * param_a + weight_b * param_b
    """
    assert isinstance(model_a, FeatureClassifier)
    assert isinstance(model_b, FeatureClassifier)
    fused = FeatureClassifier(
        in_dim=model_a.features_head[0].in_features,
        hidden_dim=model_a.features_head[0].out_features,
        bottleneck_dim=model_a.features_head[2].out_features,
        num_classes=model_a.classifier[1].out_features,
    )

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    fused_state = fused.state_dict()

    for name in fused_state.keys():
        fused_state[name] = state_a[name] * weight_a + state_b[name] * weight_b

    fused.load_state_dict(fused_state)
    return fused

