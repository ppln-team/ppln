def get_lr(optimizer):
    """Get current learning rates from runner."""
    return [group["lr"] for group in optimizer.param_groups]
