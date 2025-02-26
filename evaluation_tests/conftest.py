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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    
    # Collect test results
    total = terminalreporter._numcollected
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    # Write custom summary
    terminalreporter.write_sep("=", "Custom Test Summary")
    terminalreporter.write_line(f"Total tests: {total}")
    terminalreporter.write_line(f"Passed tests: {passed} [{passed/total:.0%}]")
    terminalreporter.write_line(f"Failed tests: {failed} [{failed/total:.0%}]")
    terminalreporter.write_line(f"Skipped tests: {skipped} [{skipped/total:.0%}]")
