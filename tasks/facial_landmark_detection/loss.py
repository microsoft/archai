def landmarks_loss(
    predicted_coords: torch.Tensor,
    label_coords: torch.Tensor) -> torch.Tensor:
    """Calculate a scalar loss value for a batch of landmark predictions, given the GT landmark labels.

    In the below:
    * B: Batch size
    * K: Number of landmarks, (aka. keypoints)

    Args:
        predicted_coords (torch.Tensor): A batch of predicted 2D landmark coordinates (B, K, 2).
        label_coords (torch.Tensor): A batch of true (GT) 2D landmark coordinates (B, K, 2).

    Returns:
        A scalar loss value, averaged over every keypoint in the batch (torch.Tensor)
    """
    assert predicted_coords.shape == label_coords.shape

    loss_fn = torch.nn.MSELoss(reduction="none")
    loss_per_landmark = loss_fn(label_coords, predicted_coords)

    return loss_per_landmark.mean()
