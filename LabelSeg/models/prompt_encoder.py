# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from LabelSeg.models.modules import LayerNorm3d

# Adapted from
# https://github.com/uni-medical/SAM-Med3D/blob/main/segment_anything/modeling/prompt_encoder3D.py  # noqa: E501


class PromptEncoder3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom3D(embed_dim // 3)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(
            self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (image_embedding_size[0],
                                image_embedding_size[1],
                                image_embedding_size[2])
        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans // 4),
            activation(),
            nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2,
                      stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            nn.Conv3d(mask_in_chans, mask_in_chans, kernel_size=2,
                      stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            nn.Conv3d(mask_in_chans, embed_dim, kernel_size=1),
        )

        self.label_embedding_net = nn.Sequential(
            nn.Linear(32, self.embed_dim))
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1xXxYxZ

    def _embed_points(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size)
        return point_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """ """
        embed = self.label_embedding_net(labels)
        return embed

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input
        prompts.
        """

        if points is not None:
            return points.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        labels: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
          labels (torch.Tensor or none): label representing the bundle
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points, with shape
            BxNx(embed_dim), where N is determined by the number of input
            points.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim),
                                        device=self._get_device())

        if points is not None:
            coords = points[:, None, :]
            point_embeddings = self._embed_points(
                coords)

            sparse_embeddings = torch.cat(
                [sparse_embeddings, point_embeddings], dim=1)

        if labels is not None:
            labels = labels[:, None, :]
            label_embeddings = self._embed_labels(
                labels)

            sparse_embeddings = torch.cat(
                [sparse_embeddings, label_embeddings], dim=1)

        dense_embeddings = self._embed_masks(masks)

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom3D(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(
        self, num_pos_feats: int = 64, scale: Optional[float] = None
    ) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2
        # shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([
            torch.sin(coords), torch.cos(coords), torch.sin(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        x, y, z = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((x, y, z), device=device)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5

        y_embed = y_embed / y
        x_embed = x_embed / x
        z_embed = z_embed / z

        pe = self._pe_encoding(torch.stack([
            x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x X x Y x Z

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords)  # B x N x C
