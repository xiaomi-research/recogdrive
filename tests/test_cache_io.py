import gzip
import pickle

import pytest

from recogdrive.cache_io import dump_feature_target_to_pickle


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
