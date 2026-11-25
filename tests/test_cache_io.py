import gzip
import pickle

import pytest

from navsim.planning.training.dataset import dump_feature_target_to_pickle


@pytest.mark.parametrize("existing", [False, True])
def test_failed_cache_write_keeps_only_complete_files(tmp_path, monkeypatch, existing):
    path = tmp_path / "features.gz"
    if existing:
        dump_feature_target_to_pickle(path, {"original": [1, 2]})

    def fail(_value, stream):
        stream.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(pickle, "dump", fail)
    with pytest.raises(OSError, match="interrupted write"):
        dump_feature_target_to_pickle(path, {"replacement": [3, 4]})

    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    if existing:
        with gzip.open(path, "rb") as stream:
            assert pickle.load(stream) == {"original": [1, 2]}


def test_resume_only_builds_missing_cache_files(tmp_path):
    from types import SimpleNamespace

    from navsim.planning.training.dataset import Dataset, load_feature_target_from_pickle

    for token in ("complete", "partial"):
        folder = tmp_path / "log" / token
        folder.mkdir(parents=True)
        dump_feature_target_to_pickle(folder / "feature.gz", {"value": token})
    dump_feature_target_to_pickle(tmp_path / "log/complete/target.gz", {"value": "complete"})
    partial_target = tmp_path / "log/partial/target.gz"
    partial_target.touch()

    def load_scene(token):
        assert token == "partial"
        return SimpleNamespace(scene_metadata=SimpleNamespace(log_name="log", initial_token=token),
                               get_agent_input=lambda: None)

    loader = SimpleNamespace(tokens=["complete", "partial"], get_scene_from_token=load_scene)
    feature = SimpleNamespace(get_unique_name=lambda: "feature",
                              compute_features=lambda _: pytest.fail("completed feature was overwritten"))
    target = SimpleNamespace(get_unique_name=lambda: "target", compute_targets=lambda _: {"value": "rebuilt"})
    Dataset(loader, [feature], [target], cache_path=str(tmp_path), force_cache_computation=False)
    assert load_feature_target_from_pickle(partial_target) == {"value": "rebuilt"}
    assert load_feature_target_from_pickle(tmp_path / "log/complete/target.gz") == {"value": "complete"}
