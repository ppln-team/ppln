def get_lr(optimizers):
    """Get current learning rates from runner."""
    return {name: [group['lr'] for group in optimizer.param_groups] for name, optimizer in optimizers.items()}
