from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Code for custom loss functions proposed across the research
# Key points: Each of these needs to have *args, to maintain cross compatability with other loss functions
# Also maintain a structure like: predictions, gtscore and then any other additional arguments


# TODO: Check if a different logic for the sequences would help it
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
    KEYFRAME_SCORES: str = "model_predictions"
    KEYFRAME_LABELS: str = "keyframe_labels"
    KEYFRAME_MASK: str = "keyframe_mask"
    ANNOTATOR_LABELS: str = "annotator_labels"

    FRAME_FEATURES: str = "frame_features"
    PREDICTED_FEATURES: str = "projections"
    optional_inputs: list[str] | None = None

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        raise NotImplementedError()

    def get_required_inputs(self) -> set[str]:
        raise NotImplementedError()


class WeightedBinaryCrossEntropy(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    def __call__(
        self,
        model_predictions: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_predictions: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy weigthed according to #keyframes/N.
        """
        pred = model_predictions.reshape(-1)  # I have to flatten this to then mask
        target = keyframe_labels.reshape(-1)  # I have to flatten this to then mask

        assert pred.shape == target.shape

        if keyframe_mask is not None:
            keyframe_mask = keyframe_mask.reshape(-1)
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]

        w_key = target.sum() / target.shape[0]
        w_back = 1 - w_key
        loss = F.binary_cross_entropy(
            pred, target, weight=torch.tensor([w_key, w_back])
        )
        return loss

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class FeatureReconstructionLoss(VideoLoss):
    def __init__(self, top_k) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]
        self._top_k = top_k

    def __call__(
        self,
        model_predictions: torch.Tensor,
        projection: torch.Tensor,
        frame_features: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            projection: individual frame features obtained through some model
            frame_features: actual observed frame features
            feature_maks: boolean inclusion mask

        Returns:
            Feature Reconstruction Loss.
        """
        pred = projection
        target = frame_features
        if len(pred.shape) == 2:
            pred = torch.unsqueeze(pred, 0)
            target = torch.unsqueeze(target, 0)
        if keyframe_mask is not None:
            model_predictions = model_predictions.masked_fill_(~keyframe_mask, -1e9)
        _, top_k_indices = torch.topk(model_predictions, self.self._top_k)
        B_idx = torch.arange(pred.shape[0], device=pred.device)[:, None]
        topk_vectors_orig = target[B_idx, top_k_indices]
        topk_vectors_recon = pred[B_idx, top_k_indices]
        assert pred.shape == target.shape

        diff = topk_vectors_orig - topk_vectors_recon
        dist = torch.linalg.norm(diff, ord=2, dim=-1)

        return dist.mean()

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.FRAME_FEATURES, self.PREDICTED_FEATURES}


class DiversityLoss(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    def __call__(
        self,
        projection: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            projection: individual frame features obtained through some model
            feature_maks: boolean inclusion mask

        Returns:
            Pairwise cosine distance.
        """
        pred = projection

        if len(pred.shape) == 2:
            pred = torch.unsqueeze(pred, 0)
        p_norm = F.normalize(pred, p=2, dim=-1)
        sim_matrix = torch.matmul(p_norm, p_norm.transpose(-2, -1))
        if keyframe_mask is not None:
            assert pred.shape[0] == keyframe_mask.shape[0]
            keyframe_mask = keyframe_mask.unsqueeze(1) & keyframe_mask.unsqueeze(2)
            sim_matrix = sim_matrix.masked_fill(~keyframe_mask, 0)

        mask = ~torch.eye(sim_matrix.shape[1], dtype=torch.bool, device=pred.device)
        m_off_diag = sim_matrix.masked_fill(~mask, 0)
        if keyframe_mask is not None:
            mask = ~keyframe_mask & mask.unsqueeze(0)
        return m_off_diag.sum() / mask.sum()

    def get_required_inputs(self) -> set[str]:
        return {self.PREDICTED_FEATURES}


class MeanSquaredError(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    def __call__(
        self,
        model_predictions: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_predictions: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy loss.
        """

        pred = model_predictions.reshape(-1)  # I have to flatten this to then mask
        target = keyframe_labels.reshape(-1)  # I have to flatten this to then mask

        assert pred.shape == target.shape

        if keyframe_mask is not None:
            keyframe_mask = keyframe_mask.reshape(-1)
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]

        return F.mse_loss(pred, target)

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class BinaryCrossEntropyMulti(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    def __call__(
        self,
        model_predictions: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            model_predictions: predicted score in [0, 1]
            keyframe_labels: ground truth label shaped B x N x D in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy loss optimized for all sequences.
        """
        pred = model_predictions
        target = keyframe_labels
        assert len(target.shape) > 2, (
            "This loss is designed to optimize multiple gt objectives at once"
        )
        repeats = target.shape[1]
        pred = pred.repeat_interleave(repeats, dim=0)
        target = target.reshape(-1)
        pred = pred.reshape(-1)
        if keyframe_mask is not None:
            keyframe_mask = keyframe_mask.repeat_interleave(repeats, dim=0)
            keyframe_mask = keyframe_mask.reshape(-1)
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]
        return F.binary_cross_entropy(pred, target)

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class MeanSquareErrorMulti(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    def __call__(
        self,
        model_predictions: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            model_predictions: predicted score in [0, 1]
            keyframe_labels: ground truth label shaped B x N x D in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy loss optimized for all sequences.
        """
        pred = model_predictions
        target = keyframe_labels
        if len(target.shape) < 2:
            target = target[None, :, :]
        repeats = target.shape[1]
        pred = pred.repeat_interleave(repeats, dim=0)
        target = target.reshape(-1)
        pred = pred.reshape(-1)
        if keyframe_mask is not None:
            keyframe_mask = keyframe_mask.repeat_interleave(repeats, dim=0)
            keyframe_mask = keyframe_mask.reshape(-1)
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]
        return F.mse_loss(pred, target)

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class BinaryCrossEntropy(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    def __call__(
        self,
        model_predictions: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_predictions: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy loss.
        """
        pred = model_predictions.reshape(-1)  # I have to flatten this to then mask
        target = keyframe_labels.reshape(-1)  # I have to flatten this to then mask

        assert pred.shape == target.shape

        if keyframe_mask is not None:
            keyframe_mask = keyframe_mask.reshape(-1)
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]

        return F.binary_cross_entropy(pred, target)

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class LengthRegularizationLoss(VideoLoss):
    def __init__(self, summary_ratio: float) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]
        self.__summary_ratio = summary_ratio

    def __call__(
        self, model_predictions: torch.Tensor, keyframe_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            model_predictions: predicted score in [0, 1]
            keyframe_mask: boolean inclusion mask

        Returns:
            Length regularization balanced by summary ratio.
        """
        pred = model_predictions
        if len(pred.shape) == 1:
            pred = torch.unsqueeze(pred, 0)
        if pred.shape[0] > 1:
            if keyframe_mask is not None:
                assert pred.shape[0] == keyframe_mask.shape[0]
                masked_pred = pred.masked_fill(~keyframe_mask, 0)
                valid_sums = keyframe_mask.sum(dim=1).clamp(min=1)
                mean_preds = masked_pred.sum(dim=1) / valid_sums
                return torch.abs(mean_preds - self.__summary_ratio).mean()

        return torch.abs(torch.mean(pred.squeeze()) - self.__summary_ratio)

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES}


class VariationLoss(
    VideoLoss
):  # This one uses an approximation as it doesn't consider all subsets. The gradient should still be somewhat similar.
    def __init__(self, beta: float) -> None:
        self.optional_inputs = [self.FEATURE_MASK, self.KEYFRAME_MASK]
        self.__beta = beta

    def __call__(
        self,
        projection: torch.Tensor,
        model_predictions: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            projection: individual frame features obtained through some model
            model_predictions: predicted score in [0, 1]
            keyframe_mask: boolean inclusion mask

        Returns:
            Variation loss.
        """
        phi = projection
        y = model_predictions

        assert phi.shape[0] == y.shape[0]

        if len(phi.shape) == 2:
            phi = torch.unsqueeze(phi, 0)
            y = torch.unsqueeze(y, 0)

        if keyframe_mask is not None:
            assert phi.shape[0] == keyframe_mask.shape[0]

            phi = phi[keyframe_mask]
            y = y[keyframe_mask]

        N = phi.size(0)

        sq_dists = torch.cdist(phi, phi, p=2).pow(2)

        Phi = torch.exp(-self.__beta * sq_dists)
        L = torch.outer(y, y) * Phi

        I = torch.eye(N, device=y.device, dtype=y.dtype)

        logdet_L = torch.logdet(L)
        logdet_L_plus_I = torch.logdet(L + I)

        L_var = -(logdet_L - logdet_L_plus_I)

        return L_var

    def get_required_inputs(self) -> set[str]:
        return {self.PREDICTED_FEATURES, self.KEYFRAME_SCORES, self.KEYFRAME_MASK}


class MAAMLoss(VideoLoss):
    lam: torch.Tensor

    def __init__(self) -> None:
        self.optional_inputs = {self.KEYFRAME_MASK}

    def __call__(
        self,
        annotator_labels: torch.Tensor,
        keyframe_scores: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = keyframe_scores
        targets = annotator_labels
        mask = keyframe_mask if keyframe_mask is not None else torch.ones_like(targets)

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        mask = mask.to(targets.dtype).expand_as(targets)

        sq = (targets - pred.unsqueeze(1)) ** 2 * mask

        cnt = mask.sum(dim=(0, 2))
        mse = sq.sum(dim=(0, 2)) / cnt.clamp_min(1)
        lam = torch.where(
            cnt > 0, 1.0 / mse.clamp_min(1e-6), torch.zeros_like(mse)
        ).detach()

        return (lam.view(1, -1, 1) * sq).sum() / mask.sum().clamp_min(1)

    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.ANNOTATOR_LABELS}


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

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        loss_values = []

        for l, w in zip(self.losses, self.weights):
            req_inputs = l.get_required_inputs()
            assert req_inputs <= kwargs.keys(), (
                "Please provide all required keyword arguments: "
                + str(self.required_inputs)
            )

            optional_keys = l.optional_inputs

            used_parameters = {k: kwargs[k] for k in req_inputs}

            if optional_keys is not None:
                for k in optional_keys:
                    if k in kwargs.keys():
                        used_parameters[k] = kwargs[k]

            loss = l(**used_parameters)
            loss_values.append(w * loss)

        return torch.stack(loss_values).sum()

    def get_required_inputs(self) -> set[str]:
        return self.required_inputs


loss_dict = {"mse": F.mse_loss, "bce": F.binary_cross_entropy}
loss_dict_stacker = {
    "weighted_bce": WeightedBinaryCrossEntropy,
    "bce": BinaryCrossEntropy,
    "feature_recon_loss": FeatureReconstructionLoss,
    "diversity": DiversityLoss,
    "mse": MeanSquaredError,
    "leng_reg": LengthRegularizationLoss,
    "bce_multi": BinaryCrossEntropyMulti,
    "mse_multi": MeanSquareErrorMulti,
}
if __name__ == "__main__":
    #    clipit_loss = WeightedLossStack(
    #        [WeightedBinaryCrossEntropy(), FeatureReconstructionLoss(), DiversityLoss()],
    #        [0.3, 0.4, 0.5],
    #    )
    #
    k_score = torch.tensor((0.2, 0.3, 0.6))
    #    k_label = torch.tensor((0.0, 0.0, 1.0, 0.0, 1.0))
    #
    #    clipit_decode = torch.tensor(((0.0, 0.0), (1.0, 2.0), (1.0, 2.0)))
    frame_features = torch.tensor(((0.0, 1.0), (1.0, 2.0), (1.0, 3.0)))
    #
    #    keyframe_mask = torch.tensor((True, True, False, False, False))
    #
    #    print(
    #        clipit_loss(
    #            keyframe_scores=k_score,
    #            keyframe_labels=k_label,
    #            keyframe_mask=keyframe_mask,
    #            clipit_decoded_features=clipit_decode,
    #            clipit_frame_features=clipit_frame,
    #        )
    #    )
    loss = VariationLoss(1)
    print(loss(frame_features, k_score))
