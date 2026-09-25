from recogdrive.eval.registry import EVALUATORS, SubprocessEvaluator, build_evaluator, register

import recogdrive.eval.navsim  # noqa: F401  (registers "navsim")
import recogdrive.eval.vqa  # noqa: F401  (registers "drivelm", "lingoqa", "drivebench")

__all__ = ["EVALUATORS", "SubprocessEvaluator", "build_evaluator", "register"]
