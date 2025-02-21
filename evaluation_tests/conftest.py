import pytest
import tensorflow as tf

from model import Word2Vec


def pytest_addoption(parser):
    parser.addoption("--model-path", action="store", default=None, help="Path to the saved model")


@pytest.fixture
def model_path(request):
    return request.config.getoption("--model-path")


@pytest.fixture
def model(model_path):
    if model_path is None:
        raise ValueError("Model path must be provided using --model-path option")
    return tf.keras.models.load_model(model_path, custom_objects={"Word2Vec": Word2Vec})
