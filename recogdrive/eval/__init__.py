from recogdrive.eval.registry import EVALUATORS, SubprocessEvaluator, build_evaluators, register

import recogdrive.eval.navsim  # noqa: F401  (registers "navsim")
import recogdrive.eval.openloop  # noqa: F401  (registers "nuscenes", "waymoe2e")
import recogdrive.eval.vqa  # noqa: F401  (registers "drivelm", "lingoqa", "drivebench")

__all__ = ["EVALUATORS", "SubprocessEvaluator", "build_evaluators", "register"]
