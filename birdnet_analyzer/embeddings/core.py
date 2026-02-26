from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
    from perch_hoplite.db import sqlite_usearch_impl

DATASET_NAME: str = "birdnet_analyzer_dataset"
COMMIT_BS_SIZE = 512
SETTINGS_KEY = "birdnet_analyzer_settings"


def embeddings(
    audio_input: str,
    database: str,
    *,
    overlap: float = 0.0,
    audio_speed: float = 1.0,
    fmin: int = 0,
    fmax: int = 15000,
    batch_size: int = 1,
    file_output: str | None = None,
    n_workers: int | None = None,
    n_producers: int = 1,
    on_update: Callable[[AcousticProgressStats], None] | None = None,
):
    """
    Generates embeddings for audio files using the BirdNET-Analyzer.
    This function processes audio files to extract embeddings, which are
    representations of audio features. The embeddings can be used for
    further analysis or comparison.
    Args:
        audio_input (str): Path to the input audio file or directory containing audio files.
        database (str): Path to the database where embeddings will be stored.
        overlap (float, optional): Overlap between consecutive audio segments in seconds. Defaults to 0.0.
        audio_speed (float, optional): Speed factor for audio processing. Defaults to 1.0.
        fmin (int, optional): Minimum frequency (in Hz) for audio analysis. Defaults to 0.
        fmax (int, optional): Maximum frequency (in Hz) for audio analysis. Defaults to 15000.
        threads (int, optional): Number of threads to use for processing. Defaults to 8.
        batch_size (int, optional): Number of audio segments to process in a single batch. Defaults to 1.
        file_output (str | None, optional): Path to save the output embeddings. If None, embeddings are not saved to a file. Defaults to None.
        n_workers (int | None, optional): Number of worker threads to use for processing. Defaults to None.
        n_producers (int, optional): Number of producer threads to use for processing. Defaults to 1.
        on_update (Callable[[AcousticProgressStats], None] | None, optional): Callback function to report progress updates. Defaults to None.
    Raises:
        FileNotFoundError: If the input path or database path does not exist.
        ValueError: If any of the parameters are invalid.
    Example:
        embeddings(
            "path/to/audio",
            "path/to/database",
            overlap=0.5,
            audio_speed=1.0,
            fmin=500,
            fmax=10000,
            threads=4,
            batch_size=2,
        )
    """
    from birdnet_analyzer.model_utils import get_embeddings

    result = get_embeddings(
        audio_input,
        version="2.4",
        batch_size=batch_size,
        overlap_duration_s=overlap,
        bandpass_fmin=fmin,
        bandpass_fmax=fmax,
        speed=audio_speed,
        n_workers=n_workers,
        n_producers=n_producers,
        callback=on_update,
    )

    batchsize = COMMIT_BS_SIZE
    batch = 0
    db = _get_or_create_database(database)
    _check_database_settings(db, fmin=fmin, fmax=fmax, audio_speed=audio_speed)

    for row in tqdm(result, desc="Saving embeddings to database"):
        if _try_consume_embedding(row["input"], row["start_time"], row["end_time"], row["embedding"], db):
            batch += 1

        if batch >= batchsize:
            db.commit()
            batch = 0

    db.commit()
    db.db.close()

    if file_output:
        create_csv_output(file_output, database)


def create_csv_output(output_path: str, database: str):
    """Creates a CSV output for the database.

    Args:
        output_path: Path to the output file.
        database: Path to the database.
    """

    db = _get_or_create_database(database)
    parent_dir = os.path.dirname(output_path)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    embedding_ids = db.get_embedding_ids()

    csv_content = "file_path,start,end,embedding\n"

    for embedding_id in embedding_ids:
        embedding = db.get_embedding(embedding_id)
        source = db.get_embedding_source(embedding_id)

        start, end = source.offsets

        csv_content += f'{source.source_id},{start},{end},"{",".join(map(str, embedding.tolist()))}"\n'

    with open(output_path, "w") as f:
        f.write(csv_content)


def _try_consume_embedding(fpath, s_start, s_end, embeddings, db: sqlite_usearch_impl.SQLiteUsearchDB, dataset_name: str = DATASET_NAME):
    import numpy as np
    from perch_hoplite.db import interface as hoplite

    existing_embedding = db.get_embeddings_by_source(dataset_name, fpath, np.array([s_start, s_end]))

    if existing_embedding.size == 0:
        embeddings_source = hoplite.EmbeddingSource(dataset_name, fpath, np.array([s_start, s_end]))

        db.insert_embedding(embeddings, embeddings_source)

        return True

    return False


def _get_or_create_database(db_path: str, embedding_dim: int = 1024):
    """Get the database object. Creates or opens the databse.
    Args:
        db: The path to the database.
    Returns:
        The database object.
    """
    import os

    from perch_hoplite.db import sqlite_usearch_impl

    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        return sqlite_usearch_impl.SQLiteUsearchDB.create(
            db_path=db_path,
            usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(embedding_dim=embedding_dim),
        )

    try:
        return sqlite_usearch_impl.SQLiteUsearchDB.create(db_path=db_path)
    except ValueError:
        return sqlite_usearch_impl.SQLiteUsearchDB.create(db_path=db_path, usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(embedding_dim=embedding_dim))


def _check_database_settings(db: sqlite_usearch_impl.SQLiteUsearchDB, fmin: int = 0, fmax: int = 15000, audio_speed: float = 1.0):
    from ml_collections import ConfigDict

    from birdnet_analyzer.embeddings.core import SETTINGS_KEY

    try:
        settings = db.get_metadata(SETTINGS_KEY)

        if settings["BANDPASS_FMIN"] != fmin or settings["BANDPASS_FMAX"] != fmax or settings["AUDIO_SPEED"] != audio_speed:
            raise ValueError(
                "Database settings do not match current configuration. DB Settings are: fmin:"
                + f"{settings['BANDPASS_FMIN']}, fmax: {settings['BANDPASS_FMAX']}, audio_speed: {settings['AUDIO_SPEED']}"
            )
    except KeyError:
        settings = ConfigDict({"BANDPASS_FMIN": fmin, "BANDPASS_FMAX": fmax, "AUDIO_SPEED": audio_speed})

        db.insert_metadata(SETTINGS_KEY, settings)
        db.commit()
