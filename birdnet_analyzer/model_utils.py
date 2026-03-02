from __future__ import annotations

from typing import TYPE_CHECKING

import birdnet

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    # from birdnet.acoustic.inference.core.encoding.encoding_result import (
    #     AcousticFileEncodingResult,
    # )  # 0.2.13
    # from birdnet.acoustic.inference.core.perf_tracker import (
    #     AcousticProgressStats,
    # )  # 0.2.13
    # from birdnet.acoustic.inference.core.prediction.prediction_result import (
    #     AcousticFilePredictionResult,
    # )  # 0.2.13
    from birdnet.acoustic_models.inference.encoding.result import (
        AcousticFileEncodingResult,
    )
    from birdnet.acoustic_models.inference.perf_tracker import AcousticProgressStats
    from birdnet.acoustic_models.inference.prediction.result import (
        AcousticFilePredictionResult,
    )
    from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_LANGUAGES

GLOBAL_PREFETCH_RATIO = 2


def run_inference(
    path,
    model="birdnet",
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
    top_k: int | None = 5,
    batch_size=1,
    n_workers: int | None = None,
    n_producers: int = 1,
    prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    overlap_duration_s=0.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    sigmoid_sensitivity=1.0,
    speed=1.0,
    min_confidence=0.1,
    custom_species_list=None,
    label_language: MODEL_LANGUAGES = "en_us",
    classifier: str | None = None,
    cc_species_list: str | None = None,
    callback: Callable[[AcousticProgressStats], None] | None = None,
) -> AcousticFilePredictionResult:
    if classifier:
        if not cc_species_list:
            cc_species_list = classifier.replace(".tflite", "_Labels.txt", 1)

        model = birdnet.load_custom(
            "acoustic", version, "tf", classifier, cc_species_list
        )
    elif model == "birdnet":
        model = birdnet.load("acoustic", version, "tf", lang=label_language)
    elif model == "perch":
        model = birdnet.load_perch_v2()

    return model.predict(
        path,
        top_k=top_k,
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        overlap_duration_s=overlap_duration_s,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        sigmoid_sensitivity=sigmoid_sensitivity,
        speed=speed,
        default_confidence_threshold=min_confidence,
        custom_species_list=custom_species_list,
        progress_callback=callback,
        show_stats="progress",
        n_workers=n_workers,
        # n_producers=n_producers, # 0.2.13
        n_feeders=n_producers,
        apply_sigmoid=False,
    )


def run_geomodel(
    lat, lon, week=None, language: MODEL_LANGUAGES = "en_us", threshold: float = 0.03
):
    model = birdnet.load("geo", "2.4", "tf", lang=language)
    return model.predict(lat, lon, week=week, min_confidence=threshold)


def get_embeddings(
    path: str,
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
    batch_size=1,
    n_workers: int | None = None,
    n_producers: int = 1,
    prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    overlap_duration_s=0.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    speed=1.0,
    callback: Callable[[AcousticProgressStats], None] | None = None,
) -> AcousticFileEncodingResult:
    model = birdnet.load("acoustic", version, "tf")
    return model.encode(
        path,
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        overlap_duration_s=overlap_duration_s,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        speed=speed,
        progress_callback=callback,
        n_workers=n_workers,
        # n_producers=n_producers, # 0.2.13
        n_feeders=n_producers,
    )


def get_embeddings_array(
    signals: list[np.ndarray],
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
    batch_size=1,
    n_workers: int | None = None,
    n_producers: int = 1,
    prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    speed=1.0,
    callback: Callable[[AcousticProgressStats], None] | None = None,
) -> np.ndarray:
    model = birdnet.load("acoustic", version, "tf")
    sr = model.get_sample_rate()

    # encode_array was removed; use encode_session + run_arrays instead.
    # run_arrays expects (ndarray, sample_rate) tuples.
    inputs = [(sig, sr) for sig in signals]

    with model.encode_session(
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        speed=speed,
        progress_callback=callback,
        n_workers=n_workers,
        # n_producers=n_producers, # 0.2.13
        n_feeders=n_producers,  # 0.2.13
    ) as session:
        result = session.run_arrays(inputs)

    # result.embeddings has shape (n_inputs, n_segments, embed_dim).
    # Each input signal is a single segment, so squeeze the middle dim.
    # Return shape: (n_inputs, embed_dim)
    return result.embeddings[:, 0, :]


def get_species_list(
    lat: float,
    lon: float,
    week: int | None,
    lang: MODEL_LANGUAGES = "en_us",
    threshold: float = 0.03,
) -> list[str]:
    model = birdnet.load("geo", "2.4", "tf", lang=lang)

    return model.predict(lat, lon, week=week, min_confidence=threshold)
