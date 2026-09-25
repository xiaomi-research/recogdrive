from recogdrive.eval.registry import EVALUATORS, SubprocessEvaluator, build_evaluator, register

import recogdrive.eval.navsim  # noqa: F401  (registers "navsim")

__all__ = ["EVALUATORS", "SubprocessEvaluator", "build_evaluator", "register"]
