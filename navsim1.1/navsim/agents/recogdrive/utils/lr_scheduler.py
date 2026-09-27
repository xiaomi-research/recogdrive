
from torch.optim.lr_scheduler import LRScheduler, _LRScheduler
import math

PROGRESS = ("last_epoch", "_step_count", "_last_lr")


class WarmupCosLR(_LRScheduler):
    """Linear warmup to `lr` over `warmup_steps`, then cosine to `min_lr` at `total_steps`, flat afterwards.
    A step is one call of step(): an optimizer step or an epoch, whichever the caller steps on."""

    def __init__(
        self, optimizer, min_lr, lr, warmup_steps, total_steps, last_epoch=-1
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, total_steps)
        super(WarmupCosLR, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Progress only: a resumed run keeps its own configuration (e.g. more epochs) from where it stopped."""
        return {key: value for key, value in self.__dict__.items() if key in PROGRESS}

    def load_state_dict(self, state_dict):
        self.__dict__.update({key: value for key, value in state_dict.items() if key in PROGRESS})

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_steps
        else:
            progress = min(1.0, (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]
        return [lr for _ in self.optimizer.param_groups]
