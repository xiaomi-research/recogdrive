import math
import unittest

import torch

from recogdrive.models.recogdrive.utils.lr_scheduler import WarmupCosLR


class RecipeCompatibilityTests(unittest.TestCase):
    def test_original_main_epoch_schedule_including_after_recipe_end(self):
        for warmup, epochs, minimum in ((3, 200, 1e-6), (0, 10, 0.0)):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([parameter], lr=1e-4)
            schedule = WarmupCosLR(optimizer, minimum, 1e-4, warmup, epochs, legacy=True)
            # This formula is from main d544047 utils/lr_scheduler.py, stepped once per epoch.
            for epoch in range(epochs + 5):
                expected = (1e-4 * (epoch + 1) / warmup if epoch < warmup else
                            minimum + 0.5 * (1e-4 - minimum) *
                            (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))))
                self.assertEqual(optimizer.param_groups[0]["lr"], expected)
                optimizer.step()
                schedule.step()


if __name__ == "__main__":
    unittest.main()
