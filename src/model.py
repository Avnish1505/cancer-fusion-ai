"""
Model builder for Phase 1 (image-only baseline).
Kept deliberately simple and swappable — Phase 2 will extend this with
a tabular branch + fusion layer, so the image encoder here is designed
to be reused as a feature extractor later (see `forward_features`).
"""
import torch
import torch.nn as nn
import torchvision.models as models


SUPPORTED_BACKBONES = ["resnet50", "efficientnet_b0"]


class CancerImageClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 7,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Supported: {SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()  # strip original classifier
            self.feature_dim = in_features

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.feature_dim = in_features

        self.encoder = base_model

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw feature embedding (useful for Phase 2 fusion)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor, meta: torch.Tensor = None) -> torch.Tensor:
        # Phase 1: image-only
        # Phase 2 will add a tabular branch and fuse its output with these features
        # if meta is not None:
        #     ...

        features = self.forward_features(x)
        logits = self.classifier_head(features)
        return logits


def build_model(config: dict) -> CancerImageClassifier:
    model_cfg = config["model"]
    model = CancerImageClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )
    return model
