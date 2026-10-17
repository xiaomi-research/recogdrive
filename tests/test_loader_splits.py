"""An unsupported test split must never silently read validation labels."""
import pytest
from omegaconf import OmegaConf

from recogdrive.data.nuscenes import nuscenes_loader
from recogdrive.data.physicalai import physicalai_loader
from recogdrive.data.vqa import vqa_loader


@pytest.mark.parametrize("loader", [nuscenes_loader, physicalai_loader, vqa_loader])
def test_unsupported_test_split_cannot_fall_back_to_val(loader):
    with pytest.raises(ValueError, match="train/val"):
        loader(OmegaConf.create({}), object(), "test")
