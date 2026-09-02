import torch


def get_mask(sdf_tensor: torch.Tensor, normalizer=None) -> torch.Tensor:
    """
    Return fluid mask from SDF using the DC-NCA convention.

    - `SDF > 0`   -> fluid (1)
    - `SDF <= 0`  -> solid (0)

    If `normalizer` is available, denormalize the SDF before thresholding.
    """
    if normalizer is not None and hasattr(normalizer, "denorm_sdf"):
        sdf_raw = normalizer.denorm_sdf(sdf_tensor)
        return (sdf_raw > 0.0).float()

    return (sdf_tensor > 0.0).float()


def get_statics(x: torch.Tensor) -> torch.Tensor:
    """
    Extract the 4 static channels from a full state tensor.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")

    if x.shape[1] < 4:
        raise ValueError(f"Expected at least 4 channels, got {x.shape[1]}")

    return x[:, 0:4, :, :]


def get_mask_from_state(
    x: torch.Tensor, threshold: float = 0.0, normalizer=None
) -> torch.Tensor:
    """
    Return fluid mask from a full state tensor where channel 0 stores SDF.
    Uses the same convention as `get_mask`: `SDF > threshold` is fluid.
    """
    sdf = x[:, 0:1] if x.ndim == 4 else x[0:1]
    if normalizer is not None and hasattr(normalizer, "denorm_sdf"):
        sdf = normalizer.denorm_sdf(sdf)
    return (sdf > threshold).float()
