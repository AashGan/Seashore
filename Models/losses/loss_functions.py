from typing import Any, override

import numpy as np
import torch
import torch.nn.functional as F


# Code for custom loss functions proposed across the research
# Key points: Each of these needs to have *args, to maintain cross compatability with other loss functions
# Also maintain a structure like: predictions, gtscore and then any other additional arguments
def masked_mse(
    predictions: torch.tensor, gtscore: torch.tensor, mask: torch.BoolTensor = None
):
    """
    Ensure style of padding is True for padded tokens, False for the not padded tokens
    """
    loss = F.mse_loss(predictions, gtscore, reduction="none")
    if mask is not None:
        mask = (~mask.bool()).expand_as(loss)
        loss = loss[mask]

    return loss.mean()


def length_regularization_loss(scores, reg_factor=0.6):
    """Source: https://github.com/e-apostolidis/CA-SUM/blob/main/model/solver.py
    Compute the summary-length regularization loss based on eq. (1).

    :param torch.Tensor scores: Frame-level importance scores, produced by our CA-SUM model.
    :return: A (torch.Tensor) value indicating the summary-length regularization loss.
    """
    return torch.abs(torch.mean(scores) - reg_factor)


def masked_soft_binary_cross_entropy(pred, target, mask=None):
    """
    Args:
        pred:   predicted probabilities in [0, 1]
        target: soft binary targets in [0, 1]
        mask:   optional boolean/0-1 mask; True/1 = include

    Returns:
        Scalar BCE loss.
    """
    loss = F.binary_cross_entropy(pred, target.float(), reduction="none")

    if mask is not None:
        mask = mask.bool()

        if mask.sum() == 0:
            return pred.new_tensor(0.0)

        loss = loss[mask]

    return loss.mean()


# TODO: loss functions which rely on selected keyframes from CLIP-IT etc


class VideoLoss:
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor():
        raise NotImplementedError()

    def get_required_inputs(self) -> set[str]:
        raise NotImplementedError()


class WeightedBinaryCrossEntropy(VideoLoss):
    @override
    def __call__(
        self, keyframe_scores: torch.Tensor, keyframe_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            keyframe_scores: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}

        Returns:
            Binary Cross Entropy weigthed according to #keyframes/N.
        """
        pred = keyframe_scores
        target = keyframe_labels

        assert pred.shape == target.shape
        w = target.sum() / target.shape[0]

        keyframe_component = w * target * pred.log()
        background_component = (1 - w) * (1 - target) * (1 - pred).log()

        return -(keyframe_component + background_component).mean()

    @override
    def get_required_inputs(self) -> set[str]:
        return {"keyframe_scores", "keyframe_labels"}


class FeatureReconstructionLoss(VideoLoss):
    @override
    def __call__(
        self, clipit_decoded_features: torch.Tensor, clipit_frame_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            clipit_decoded_features: decoded features from CLIP-IT transformer output
            clipit_frame_features: CLIP-IT frame features

        Returns:
            Feature Reconstruction Loss.
        """
        pred = clipit_decoded_features
        target = clipit_frame_features

        assert pred.shape == target.shape
        diff = target - pred
        dist = torch.linalg.norm(diff, ord=2, dim=-1)

        return dist.mean()

    @override
    def get_required_inputs(self) -> set[str]:
        return {"clipit_decoded_features", "clipit_frame_features"}


class DiversityLoss(VideoLoss):
    @override
    def __call__(self, clipit_decoded_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clipit_decoded_features: decoded features from CLIP-IT transformer output

        Returns:
            Pairwise cosine distance.
        """

        p_norm = F.normalize(clipit_decoded_features, p=2, dim=-1)
        sim_matrix = torch.mm(p_norm, p_norm.t())

        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
        m_off_diag = sim_matrix * mask

        return m_off_diag.sum() / mask.sum()

    @override
    def get_required_inputs(self) -> set[str]:
        return {"clipit_decoded_features"}


class WeightedLossStack(VideoLoss):
    losses: list[VideoLoss]
    weights: list[float]
    required_inputs: set[str]

    def __init__(self, losses: list[VideoLoss], weights: list[float]) -> None:
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights

        self.required_inputs = set().union(
            *[l.get_required_inputs() for l in self.losses]
        )

    @override
    def __call__(self, **kwargs: Any) -> torch.Tensor():
        loss_values = []

        for l, w in zip(self.losses, self.weights):
            req_inputs = l.get_required_inputs()
            assert req_inputs <= kwargs.keys()

            used_parameters = {k: kwargs[k] for k in req_inputs}
            loss = l(**used_parameters)
            loss_values.append(w * loss)

        return torch.tensor(loss_values).sum()

    @override
    def get_required_inputs(self) -> set[str]:
        return self.required_inputs


loss_dict = {"mse": F.mse_loss, "bce": F.binary_cross_entropy}

if __name__ == "__main__":
    clipit_loss = WeightedLossStack(
        [WeightedBinaryCrossEntropy(), FeatureReconstructionLoss(), DiversityLoss()],
        [0.3, 0.4, 0.5],
    )

    k_score = torch.tensor((0.2, 0.3, 0.6, 0.4, 0.8))
    k_label = torch.tensor((0.0, 0.0, 1.0, 0.0, 1.0))

    clipit_decode = torch.tensor(((0.0, 0.0), (1.0, 2.0), (1.0, 2.0)))
    clipit_frame = torch.tensor(((0.0, 1.0), (1.0, 2.0), (1.0, 3.0)))

    print(
        clipit_loss(
            keyframe_scores=k_score,
            keyframe_labels=k_label,
            clipit_decoded_features=clipit_decode,
            clipit_frame_features=clipit_frame,
        )
    )

