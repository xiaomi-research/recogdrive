"""Write feature and target caches without exposing unfinished files."""

import gzip
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile


def dump_feature_target_to_pickle(path: Path, data_dict: dict) -> None:
    path = Path(path)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=temporary, compresslevel=1) as archive:
                pickle.dump(data_dict, archive)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
