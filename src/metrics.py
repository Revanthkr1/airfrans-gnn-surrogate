"""Evaluation metrics. Always report relative L2 per field -- never one blended number."""
FIELD_NAMES = ["vx", "vy", "pressure", "nu_t"]


def relative_l2_per_field(pred, target, eps=1e-8):
    """pred, target: (N, 4) tensors in the SAME units (de-normalized for reporting).

    Returns dict {field_name: relative L2 error} computed over all rows given.
    Not meaningful if `target`'s true norm is near zero for a field over the
    given rows (e.g. velocity restricted to wall/surface nodes, which is ~0
    there by the no-slip condition) -- use mean_abs_error_per_field instead.
    """
    errors = {}
    for i, field in enumerate(FIELD_NAMES):
        num = (pred[:, i] - target[:, i]).norm()
        den = target[:, i].norm() + eps
        errors[field] = (num / den).item()
    return errors


def mean_abs_error_per_field(pred, target):
    """pred, target: (N, 4) tensors in the SAME units.

    Absolute, not relative -- for regions where the true value can be
    legitimately near zero (e.g. velocity at wall/surface nodes), where
    relative_l2_per_field's division blows up numerically even for a tiny
    absolute error.
    """
    errors = {}
    for i, field in enumerate(FIELD_NAMES):
        errors[field] = (pred[:, i] - target[:, i]).abs().mean().item()
    return errors
