"""DINOv2 ViT-B/14 features decoded into 64 square predictions by learned square queries."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from .board import CLASSES

IMAGE_SIZE = 644  # 46 x 46 patches of 14 px
ENCODER = "vit_base_patch14_dinov2.lvd142m"
WEIGHTS_REPO = "timm/vit_base_patch14_dinov2.lvd142m"
WEIGHTS_REVISION = "4685c99dabffe5affac90bd99dbffd25801ae58d"
WEIGHTS_SHA256 = "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b"
FUSED_BLOCKS = (2, 5, 8, 11)  # blocks 3, 6, 9 and 12 of the 12-block encoder
WIDTH = 256


def pretrained_file() -> str:
    """Download the pinned DINOv2 checkpoint and verify its digest."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WEIGHTS_REPO, "model.safetensors", revision=WEIGHTS_REVISION)
    with Path(path).open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise ValueError("DINOv2 checkpoint digest differs from the pinned release")
    return path


class FENFormer(nn.Module):
    """Map RGB images in [0, 1], shape [B, 3, 644, 644], to logits [B, 64, 13].

    Each of the 64 decoder queries is tied to one board square, so square identity is
    supervised directly and no matching step is needed. A query is the sum of a learned
    per-square embedding and shared rank, file and square-color embeddings.
    """

    def __init__(self, *, pretrained: bool = False, image_size: int = IMAGE_SIZE):
        """Build the encoder, multi-depth feature projection and square-query decoder."""
        super().__init__()
        if image_size % 14:
            raise ValueError("Image size must be a multiple of the 14 px patch size")
        import timm

        self.image_size = image_size
        overlay = {"file": pretrained_file()} if pretrained else None
        self.encoder = timm.create_model(
            ENCODER,
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,
            pretrained_cfg_overlay=overlay,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.projection = nn.Linear(self.encoder.num_features * len(FUSED_BLOCKS), WIDTH)

        self.square_queries = nn.Embedding(64, WIDTH)
        self.rank_queries = nn.Embedding(8, WIDTH)
        self.file_queries = nn.Embedding(8, WIDTH)
        self.color_queries = nn.Embedding(2, WIDTH)
        squares = torch.arange(64)
        ranks, files = 7 - squares // 8, squares % 8  # FEN order starts at a8
        self.register_buffer("rank_index", ranks)
        self.register_buffer("file_index", files)
        self.register_buffer("color_index", (ranks + files + 1) % 2)  # 1 = dark; a1 is dark
        for embedding in (
            self.square_queries,
            self.rank_queries,
            self.file_queries,
            self.color_queries,
        ):
            nn.init.normal_(embedding.weight, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=WIDTH,
            nhead=8,
            dim_feedforward=4 * WIDTH,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=4, norm=nn.LayerNorm(WIDTH))
        # TransformerDecoder deep-copies its prototype; re-initialize each copy independently.
        for parameter in self.decoder.layers.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        self.classifier = nn.Linear(WIDTH, len(CLASSES))

    def queries(self) -> torch.Tensor:
        """Return the 64 square queries, [64, WIDTH]."""
        return (
            self.square_queries.weight
            + self.rank_queries(self.rank_index)
            + self.file_queries(self.file_index)
            + self.color_queries(self.color_index)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode once, concatenate four depths per patch, and decode square logits."""
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected images [B, 3, {self.image_size}, {self.image_size}]")
        features = self.encoder.forward_intermediates(
            (images - self.mean) / self.std,
            indices=FUSED_BLOCKS,
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        # Concatenate along channels: [B, 2116, 4 * 768] at 644 px; prefix tokens are dropped.
        memory = self.projection(torch.cat(features, dim=-1))
        queries = self.queries().expand(images.shape[0], -1, -1)
        return self.classifier(self.decoder(queries, memory))
