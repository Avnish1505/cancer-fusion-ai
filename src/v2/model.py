"""
v2 fusion model: any timm backbone + optional metadata branch.

The metadata branch is gated: its contribution is scaled by a learned scalar
initialised near zero, so early training is driven by the image and metadata
has to earn its weight. With use_metadata=False the same class is the
image-only ablation, so the two can be compared with everything else equal.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("v2 needs timm: pip install timm") from e


class FusionNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_metadata: bool = True,
        metadata_dim: int = 0,
        meta_hidden: int = 64,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.use_metadata = use_metadata
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        self.feature_dim = self.encoder.num_features

        head_in = self.feature_dim
        if use_metadata:
            if metadata_dim <= 0:
                raise ValueError("metadata_dim must be positive when use_metadata is True")
            self.metadata_processor = nn.Sequential(
                nn.Linear(metadata_dim, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, meta_hidden),
                nn.SiLU(inplace=True),
            )
            self.meta_gate = nn.Parameter(torch.tensor(0.1))
            head_in += meta_hidden
        else:
            self.metadata_processor = None

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled image embedding (image branch only). Used by the OOD guard."""
        return self.encoder(x)

    def head(self, image_features: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        if self.use_metadata:
            if meta is None:
                raise ValueError("This model was built with use_metadata=True; pass a metadata tensor.")
            meta_features = self.metadata_processor(meta) * self.meta_gate
            image_features = torch.cat((image_features, meta_features), dim=1)
        return self.classifier_head(image_features)

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x), meta)

    def param_groups(self, lr: float, head_lr_mult: float, weight_decay: float) -> list[dict]:
        """Backbone at `lr`, freshly initialised parts at `lr * head_lr_mult`.
        Norm/bias params get no weight decay."""
        groups = {k: {"params": [], "lr": lr * m, "weight_decay": wd}
                  for k, m, wd in [("enc", 1.0, weight_decay), ("enc_nd", 1.0, 0.0),
                                   ("new", head_lr_mult, weight_decay), ("new_nd", head_lr_mult, 0.0)]}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            base = "enc" if name.startswith("encoder.") else "new"
            no_decay = p.ndim <= 1
            groups[base + ("_nd" if no_decay else "")]["params"].append(p)
        return [g for g in groups.values() if g["params"]]


def build_fusion_model(model_cfg: dict, use_metadata: bool, metadata_dim: int, pretrained: bool | None = None) -> FusionNet:
    return FusionNet(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg.get("num_classes", 7),
        pretrained=model_cfg.get("pretrained", True) if pretrained is None else pretrained,
        dropout=model_cfg.get("dropout", 0.3),
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    )
