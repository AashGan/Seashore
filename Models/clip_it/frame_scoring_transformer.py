import torch
import torch.nn as nn


class FrameScoringTransformer(nn.Module):
    transformer: nn.Transformer
    score_head: nn.Linear

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.score_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape [n_batches, n_frames, n_frame_features]
            frame_mask: tensor of shape [n_batches, n_frames]
            return_features: include features in function return

        Returns:
            Tensor of shape [n_batches, n_frames] describing the score of each frame, as well as the output features of the transformer if return_features is True.

        """
        if frame_mask is not None:
            pad = ~frame_mask
        else:
            pad = None
        h = self.transformer(
            src=x,
            tgt=x,
            src_key_padding_mask=pad,
            tgt_key_padding_mask=pad,
            memory_key_padding_mask=pad,
        )
        logits = self.score_head(h).squeeze(-1)
        return (logits, h) if return_features else logits
