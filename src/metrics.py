"""Evaluation metrics. Always report relative L2 per field -- never one blended number."""
FIELD_NAMES = ["vx", "vy", "pressure", "nu_t"]


def relative_l2_per_field(pred, target, eps=1e-8):
    """pred, target: (N, 4) tensors in the SAME units (de-normalized for reporting).

    Returns dict {field_name: relative L2 error} computed over all rows given.
    """
    errors = {}
    for i, field in enumerate(FIELD_NAMES):
        num = (pred[:, i] - target[:, i]).norm()
        den = target[:, i].norm() + eps
        errors[field] = (num / den).item()
    return errors
