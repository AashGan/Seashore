import torch
import torch.nn as nn

from .frame_scoring_transformer import FrameScoringTransformer
from .language_guided_attention import LanguageGuidedAttention


class CLIP_IT(nn.Module):
    lg_attention: LanguageGuidedAttention
    scoring_transformer: FrameScoringTransformer

    def __init__(
        self,
        n_frame_features: int,
        n_text_features: int,
        n_head_language: int,
        n_head_transformer: int,
        n_transformer_layers: int = 6,
        transformer_droput: float = 0.1,
    ) -> None:
        super().__init__()

        self.lg_attention = LanguageGuidedAttention(
            n_frame_features, n_text_features, n_head_language
        )
        self.scoring_transformer = FrameScoringTransformer(
            n_frame_features,
            n_head_transformer,
            n_transformer_layers,
            transformer_droput,
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape [n_batches, n_frames, n_frame_features]
            text_embed: tensor of shape [n_batches, n_sentences, n_text_features]
            frame_mask: boolean inclusion mask of shape [n_batches, n_frames]
            text_mask: boolean inclusion mask of shape[n_batches, n_sentences]
            return_features: whether or not to include the final features in the output

        Returns:
            Score tensor of shape [n_batches, n_frames], and final scoring features if return_features is True.

        """

        if frame_mask is not None and text_mask is not None:
            mask = frame_mask[:, :, None] & text_mask[:, None, :]
            mask = mask.unsqueeze(1)
        elif frame_mask is not None:
            batch_size, n_sentences, _ = text_embed.size()
            text_mask = torch.ones(batch_size, n_sentences)
            mask = frame_mask[:, :, None] & text_mask[:, None, :]
            mask = mask.unsqueeze(1)
        elif text_mask is not None:
            batch_size, n_frames, _ = x.size()
            frame_mask = torch.ones(batch_size, n_frames)
            mask = frame_mask[:, :, None] & text_mask[:, None, :]
            mask = mask.unsqueeze(1)
        else:
            mask = None

        attended_features = self.lg_attention.forward(x, text_embed, mask)

        output = self.scoring_transformer.forward(
            attended_features, frame_mask=frame_mask, return_features=return_features
        )

        if return_features:
            score, features = output
            return score, features
        else:
            return output
