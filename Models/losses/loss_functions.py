import torch
import numpy as np
import torch.nn.functional as F

# Code for custom loss functions proposed across the research
# Key points: Each of these needs to have *args, to maintain cross compatability with other loss functions 
# Also maintain a structure like: predictions, gtscore and then any other additional arguments
def masked_mse(predictions:torch.tensor,gtscore:torch.tensor,mask:torch.BoolTensor = None):
    """
    Ensure style of padding is True for padded tokens, False for the not padded tokens
    """
    loss = F.mse_loss(predictions, gtscore, reduction="none")
    if mask is not None:
        mask = (~mask.bool()).expand_as(loss)
        loss = loss[mask]

    return loss.mean()



def length_regularization_loss(scores,reg_factor=0.6):
        """ Source: https://github.com/e-apostolidis/CA-SUM/blob/main/model/solver.py
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
    loss = F.binary_cross_entropy(
        pred,
        target.float(),
        reduction="none"
    )

    if mask is not None:
        mask = mask.bool()

        if mask.sum() == 0:
            return pred.new_tensor(0.0)

        loss = loss[mask]

    return loss.mean()

#TODO: loss functions which rely on selected keyframes from CLIP-IT etc

loss_dict = {'mse':F.mse_loss,'bce':F.binary_cross_entropy}