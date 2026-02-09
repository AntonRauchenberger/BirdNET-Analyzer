import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from birdnet_analyzer.cli import train_parser
from birdnet_analyzer.train.core import train


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    classifier_output = os.path.join(output_dir, "classifier_output")

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "classifier_output": classifier_output,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.train.utils.train_model")
def test_train_cli(mock_train_model, setup_test_environment):
    env = setup_test_environment

    parser = train_parser()
    args = parser.parse_args([env["input_dir"], "--output", env["classifier_output"]])

    # Remove CLI-only args not accepted by train()
    kwargs = vars(args)
    kwargs.pop("cache_mode", None)
    kwargs.pop("cache_file", None)

    train(**kwargs)

    mock_train_model.assert_called_once()
    call_kwargs = mock_train_model.call_args[1]
    assert call_kwargs["output"] == env["classifier_output"]
    assert mock_train_model.call_args[0][0] == env["input_dir"]
