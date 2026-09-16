from typing import Any, Optional, override

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
    KEYFRAME_SCORES: str = "keyframe_scores"
    KEYFRAME_LABELS: str = "keyframe_labels"
    KEYFRAME_MASK: str = "keyframe_mask"

    FRAME_FEATURES: str = "frame_features"
    PREDICTED_FEATURES: str = "predicted_features"
    FEATURE_MASK: str = "feature_mask"

    optional_inputs: list[str] | None = None

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        raise NotImplementedError()

    def get_required_inputs(self) -> set[str]:
        raise NotImplementedError()


class WeightedBinaryCrossEntropy(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    @override
    def __call__(
        self,
        keyframe_scores: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            keyframe_scores: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy weigthed according to #keyframes/N.
        """
        pred = keyframe_scores
        target = keyframe_labels

        assert pred.shape == target.shape

        if keyframe_mask is not None:
            assert keyframe_mask.shape[0] == pred.shape[0]
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]

        w = target.sum() / target.shape[0]

        keyframe_component = w * target * pred.log()
        background_component = (1 - w) * (1 - target) * (1 - pred).log()

        return -(keyframe_component + background_component).mean()

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class FeatureReconstructionLoss(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.FEATURE_MASK]

    @override
    def __call__(
        self,
        predicted_features: torch.Tensor,
        frame_features: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted_features: individual frame features obtained through some model
            frame_features: actual observed frame features
            feature_maks: boolean inclusion mask

        Returns:
            Feature Reconstruction Loss.
        """
        pred = predicted_features
        target = frame_features

        assert pred.shape == target.shape

        if feature_mask is not None:
            assert pred.shape[0] == feature_mask.shape[0]

            pred = pred[feature_mask]
            target = target[feature_mask]

        diff = target - pred
        dist = torch.linalg.norm(diff, ord=2, dim=-1)

        return dist.mean()

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.FRAME_FEATURES, self.PREDICTED_FEATURES}


class DiversityLoss(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.FEATURE_MASK]

    @override
    def __call__(
        self,
        predicted_features: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted_features: individual frame features obtained through some model
            feature_maks: boolean inclusion mask

        Returns:
            Pairwise cosine distance.
        """
        pred = predicted_features

        if feature_mask is not None:
            assert pred.shape[0] == feature_mask.shape[0]

            pred = pred[feature_mask]

        p_norm = F.normalize(pred, p=2, dim=-1)
        sim_matrix = torch.mm(p_norm, p_norm.t())

        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=pred.device)
        m_off_diag = sim_matrix * mask

        return m_off_diag.sum() / mask.sum()

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.PREDICTED_FEATURES}


class MeanSquaredError(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    @override
    def __call__(
        self,
        keyframe_scores: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            keyframe_scores: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy loss.
        """
        pred = keyframe_scores
        target = keyframe_labels

        assert pred.shape == target.shape

        if keyframe_mask is not None:
            assert pred.shape[0] == keyframe_mask.shape[0]

            pred = pred[keyframe_mask]
            target = target[keyframe_mask]

        error = pred - target
        sq_error = torch.square(error)

        return sq_error.mean()

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class BinaryCrossEntropy(VideoLoss):
    def __init__(self) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]

    @override
    def __call__(
        self,
        keyframe_scores: torch.Tensor,
        keyframe_labels: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            keyframe_scores: predicted score in [0, 1]
            keyframe_labels: ground truth label in {0, 1}
            keyframe_mask: boolean inclusion mask

        Returns:
            Binary Cross Entropy loss.
        """
        pred = keyframe_scores
        target = keyframe_labels

        assert pred.shape == target.shape

        if keyframe_mask is not None:
            assert keyframe_mask.shape[0] == pred.shape[0]
            pred = pred[keyframe_mask]
            target = target[keyframe_mask]

        keyframe_component = target * pred.log()
        background_component = (1 - target) * (1 - pred).log()

        return -(keyframe_component + background_component).mean()

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES, self.KEYFRAME_LABELS}


class LengthRegularizationLoss(VideoLoss):
    def __init__(self, summary_ratio: float) -> None:
        self.optional_inputs = [self.KEYFRAME_MASK]
        self.__summary_ratio = summary_ratio

    @override
    def __call__(
        self, keyframe_scores: torch.Tensor, keyframe_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            keyframe_scores: predicted score in [0, 1]
            keyframe_mask: boolean inclusion mask

        Returns:
            Length regularization balanced by summary ratio.
        """
        pred = keyframe_scores

        if keyframe_mask is not None:
            assert pred.shape[0] == keyframe_mask.shape[0]

            pred = pred[keyframe_mask]

        norm_diff = (pred - self.__summary_ratio) / pred.shape[0]

        dist = torch.linalg.norm(norm_diff, ord=2)

        return dist

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.KEYFRAME_SCORES}


class VariationLoss(
    VideoLoss
):  # This one uses an approximation as it doesn't consider all subsets. The gradient should still be somewhat similar.
    def __init__(self, beta: float) -> None:
        self.optional_inputs = [self.FEATURE_MASK, self.KEYFRAME_MASK]
        self.__beta = beta

    @override
    def __call__(
        self,
        predicted_features: torch.Tensor,
        keyframe_scores: torch.Tensor,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted_features: individual frame features obtained through some model
            keyframe_scores: predicted score in [0, 1]
            keyframe_mask: boolean inclusion mask

        Returns:
            Variation loss.
        """
        phi = predicted_features
        y = keyframe_scores

        assert phi.shape[0] == y.shape[0]

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

    @override
    def get_required_inputs(self) -> set[str]:
        return {self.PREDICTED_FEATURES, self.KEYFRAME_SCORES, self.KEYFRAME_MASK}


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

        return torch.tensor(loss_values).sum()

    @override
    def get_required_inputs(self) -> set[str]:
        return self.required_inputs


loss_dict = {"mse": F.mse_loss, "bce": F.binary_cross_entropy}

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
