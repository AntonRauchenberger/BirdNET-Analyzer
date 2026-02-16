import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from birdnet_analyzer.cli import embeddings_parser
from birdnet_analyzer.embeddings.core import embeddings


def _make_empty_encoding_result():
    """Create a mock AcousticFileEncodingResult with zero inputs."""
    mock_result = MagicMock()
    mock_result.segment_duration_s = 3.0
    mock_result.overlap_duration_s = 0.0
    mock_result.n_inputs = 0
    mock_result.max_n_segments = 0
    mock_result.embeddings = np.zeros((0, 0, 1024), dtype=np.float32)
    mock_result.embeddings_masked = np.zeros((0, 0, 1024), dtype=bool)
    mock_result.inputs = np.array([], dtype="<U1")
    mock_result.input_durations = np.array([])
    mock_result.input_durations = np.array([])
    return mock_result


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.embeddings.core.create_csv_output")
@patch("birdnet_analyzer.embeddings.core._check_database_settings")
@patch("birdnet_analyzer.embeddings.core._get_or_create_database")
@patch("birdnet_analyzer.model_utils.get_embeddings")
def test_embeddings_cli(
    mock_get_embeddings: MagicMock,
    mock_get_db: MagicMock,
    mock_check_settings: MagicMock,
    mock_csv_output: MagicMock,
    setup_test_environment,
):
    env = setup_test_environment

    mock_get_embeddings.return_value = _make_empty_encoding_result()
    mock_db = MagicMock()
    mock_get_db.return_value = mock_db

    parser = embeddings_parser()
    args = parser.parse_args(["--input", env["input_dir"], "-db", env["output_dir"]])

    embeddings(**vars(args))

    mock_get_embeddings.assert_called_once()
    call_kwargs = mock_get_embeddings.call_args
    assert call_kwargs[0][0] == env["input_dir"]
    assert call_kwargs[1]["version"] == "2.4"
