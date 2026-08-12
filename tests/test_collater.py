from Data.Loaders.collators import batch_collate_fn
import torch

def test_collate_fn_pads_sequences_and_creates_mask():
    batch = [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([10.0, 20.0]),
            "sample_1",
        ),
        (
            torch.tensor(
                [[5.0, 6.0],
                 [7.0, 8.0],
                 [9.0, 10.0],
                 [11.0, 12.0]]
            ),
            torch.tensor([30.0, 40.0, 50.0, 60.0]),
            "sample_2",
        ),
        (
            torch.tensor([[13.0, 14.0]]),
            torch.tensor([70.0]),
            "sample_3",
        ),
    ]

    features, gtscore, mask, data_point = batch_collate_fn(batch)

    # Longest sequence has length 4.
    assert features.shape == (3, 4, 2)
    assert gtscore.shape == (3, 4)
    assert mask.shape == (3, 4)

    # Features should be padded with 0.
    expected_features = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        [[13.0, 14.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])

    assert torch.equal(features, expected_features)

    # gtscore should be padded with -1.
    expected_gtscore = torch.tensor([
        [10.0, 20.0, -1.0, -1.0],
        [30.0, 40.0, 50.0, 60.0],
        [70.0, -1.0, -1.0, -1.0],
    ])

    assert torch.equal(gtscore, expected_gtscore)

    # Mask should identify valid (non-padded) gtscores.
    expected_mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ])

    assert torch.equal(mask, expected_mask)

    # Check that the mask agrees with the -1 padding.
    assert torch.equal(mask, gtscore != -1)

    # data_point should be preserved in the original order.
    assert data_point == ("sample_1", "sample_2", "sample_3")

