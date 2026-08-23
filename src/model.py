"""
Model builder for Phase 1 (image-only baseline).
Kept deliberately simple and swappable — Phase 2 will extend this with
a tabular branch + fusion layer, so the image encoder here is designed
to be reused as a feature extractor later (see `forward_features`).
"""
import torch
from torch import nn
from torchvision import models

SUPPORTED_BACKBONES = ["resnet50", "efficientnet_b0"]


class CancerImageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_metadata: bool = False,
        metadata_dim: int = 0,
    ):
        super().__init__()

        # --- 1. Image Backbone (Feature Extractor) ---
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

        # --- 2. Metadata Processor (Tabular MLP) ---
        self.use_metadata = use_metadata
        self.metadata_processor = None
        metadata_feature_dim = 0

        if self.use_metadata:
            if metadata_dim <= 0:
                raise ValueError("metadata_dim must be positive when use_metadata is True")
            
            # A simple 2-layer MLP to process the tabular data
            metadata_feature_dim = 64 # The output size of our metadata MLP
            self.metadata_processor = nn.Sequential(
                nn.Linear(metadata_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, metadata_feature_dim),
                nn.ReLU(inplace=True),
            )

        # --- 3. Classifier Head (takes FUSED features as input) ---
        classifier_input_dim = self.feature_dim + metadata_feature_dim

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw feature embedding from the image backbone."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor, meta: torch.Tensor = None) -> torch.Tensor:
        # 1. Get image features from the CNN backbone
        image_features = self.forward_features(x)

        if self.use_metadata:
            # 2. Get metadata features from the tabular MLP
            meta_features = self.metadata_processor(meta)
            # 3. Fuse by concatenating along the feature dimension
            fused_features = torch.cat((image_features, meta_features), dim=1)
            logits = self.classifier_head(fused_features)
        else:
            logits = self.classifier_head(image_features)
            
        return logits


def build_model(config: dict) -> CancerImageClassifier:
    model_cfg = config["model"]
    train_cfg = config["train"]

    model = CancerImageClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", False),
        dropout=model_cfg.get("dropout", 0.3),
        use_metadata=train_cfg.get("use_metadata", False),
        metadata_dim=model_cfg.get("metadata_dim", 0),
    )
    return model
