import hashlib

import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.models.recogdrive.backbone import planning_system_prompt


def test_default_planning_prompt_keeps_main_text():
    assert hashlib.sha256(planning_system_prompt().encode()).hexdigest() == (
        "0283b83f6f2c9a4eeba82b38c0682e96e99ebd93f09811a5deac28da762c0ce2"
    )


@pytest.mark.parametrize("interval", [0.25, 0.5])
def test_extended_planning_prompt_uses_horizon_and_interval(interval):
    prompt = planning_system_prompt(5, interval)
    assert "5-second trajectory" in prompt and f"{interval:g}-second intervals" in prompt
    assert "4-second" not in prompt


@pytest.mark.parametrize("cache_mode", [False, True])
def test_hidden_states_cannot_be_used_or_built_for_five_seconds(cache_mode):
    sampling = TrajectorySampling(time_horizon=5, interval_length=0.5)
    with pytest.raises(ValueError, match="cache_hidden_state=False"):
        ReCogDriveAgent(sampling, cache_hidden_state=True, cache_mode=cache_mode)
