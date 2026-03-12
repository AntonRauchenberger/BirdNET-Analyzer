"""Module for predicting a species list.

Can be used to predict a species list using coordinates and weeks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from birdnet_analyzer import model_utils

if TYPE_CHECKING:
    import numpy as np
    from birdnet.globals import MODEL_LANGUAGES
    from numpy.typing import NDArray


def get_species_list(
    lat: float, lon: float, week: int, threshold: float, lang: MODEL_LANGUAGES
) -> NDArray[np.str_]:
    """
    Generates a species list for a given location and time, and saves it to the
    specified output path.
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        week (int): Week of the year (1-52) for which the species list is generated.
        threshold (float): Threshold for location filtering.
        lang (MODEL_LANGUAGES): Language code for species names.
    Returns:
        NDArray[np.str_]: Species list as numpy strings.
    """
    result = model_utils.run_geomodel(
        lat, lon, week, threshold=threshold, language=lang
    )

    return [str(species) for species, prob in result.to_structured_array()]

