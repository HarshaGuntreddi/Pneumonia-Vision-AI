import pytest
import yaml
import numpy as np
from src.data_loader import get_data_generators

# A fixture to create a dummy directory structure for testing
@pytest.fixture
def dummy_data_dir(tmp_path):
    # Create train/val/test directories
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            d = tmp_path / "dummy_data" / split / label
            d.mkdir(parents=True)
            # Create a dummy image file
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(d / f"dummy_{label}.jpeg"), dummy_image)
    return str(tmp_path / "dummy_data")

def test_get_data_generators(dummy_data_dir):
    """
    Tests the data generator creation and batch output.
    """
    # Create a dummy config
    config = {
        'data_dir': dummy_data_dir,
        'image_size': [224, 224],
        'batch_size': 2
    }
    
    train_gen, val_gen, test_gen = get_data_generators(config)

    # Check if generators are not None
    assert train_gen is not None
    assert val_gen is not None
    assert test_gen is not None

    # Get one batch from the training generator
    x_batch, y_batch = next(train_gen)

    # Check batch shapes
    assert x_batch.shape == (2, 224, 224, 3) # (batch_size, height, width, channels)
    assert y_batch.shape == (2,) # (batch_size,)

    # Check data types
    assert x_batch.dtype == 'float32'
    assert y_batch.dtype == 'float32'