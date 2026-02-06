import os
from typing import Literal

from birdnet.globals import MODEL_LANGUAGE_EN_US, MODEL_LANGUAGES

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

#################
# Misc settings #
#################

# Random seed for gaussian noise
RANDOM_SEED: int = 42

##########################
# Model paths and config #
##########################

MODEL_VERSION: str = "V2.4"
BIRDNET_MODEL_PATH: str = os.path.join(SCRIPT_DIR, "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
BIRDNET_LABELS_FILE: str = os.path.join(SCRIPT_DIR, "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt")
TRANSLATED_LABELS_PATH: str = os.path.join(SCRIPT_DIR, "labels/V2.4")

##################
# Audio settings #
##################

# We use a sample rate of 48kHz, so the model input size is
# (batch size, 48000 kHz * 3 seconds) = (1, 144000)
# Recordings will be resampled automatically.
BIRDNET_SAMPLE_RATE: int = 48000

# We're using 3-second chunks
BIRDNET_SIG_LENGTH: float = 3.0

# Define overlap between consecutive chunks <3.0; 0 = no overlap
SIG_OVERLAP: float = 0.0

# Define minimum length of audio chunk for prediction,
# chunks shorter than 3 seconds will be padded with zeros
SIG_MINLEN: float = 1.0

# Frequency range. This is model specific and should not be changed.
SIG_FMIN: int = 0
SIG_FMAX: int = 15000

# Settings for bandpass filter
BANDPASS_FMIN: int = 0
BANDPASS_FMAX: int = 15000

# Audio speed
AUDIO_SPEED: float = 1.0

###################
# Search settings #
###################

SCORE_FUNCTIONS = Literal["cosine", "euclidean", "dot"]
CROP_MODES = Literal["center", "first", "segments"]

######################
# Inference settings #
######################

# If None or empty file, no custom species list will be used
# Note: Entries in this list have to match entries from the LABELS_FILE
# We use the 2024 eBird taxonomy for species names (Clements list)
CODES_FILE: str = os.path.join(SCRIPT_DIR, "eBird_taxonomy_codes_2024E.json")

# Supported file types
ALLOWED_FILETYPES: list[str] = ["wav", "flac", "mp3", "ogg", "m4a", "wma", "aiff", "aif"]

# Whether to use noise to pad the signal
# If set to False, the signal will be padded with zeros
USE_NOISE: bool = False

# Specifies the output format. 'table' denotes a Raven selection table,
# 'audacity' denotes a TXT file with the same format as Audacity timeline labels
# 'csv' denotes a generic CSV file with start, end, species and confidence.
RESULT_TYPES = Literal["table", "audacity", "kaleidoscope", "csv"]
ADDITIONAL_COLUMNS = Literal["lat", "lon", "week", "overlap", "sensitivity", "min_conf", "species_list", "model"]
OUTPUT_RAVEN_FILENAME: str = "BirdNET_SelectionTable.txt"  # this is for combined Raven selection tables only
OUTPUT_KALEIDOSCOPE_FILENAME: str = "BirdNET_Kaleidoscope.csv"
OUTPUT_CSV_FILENAME: str = "BirdNET_CombinedTable.csv"
OUTPUT_AUDACITY_FILENAME: str = "BirdNET_AudacityLabels.txt"

# File name of the settings csv for batch analysis
ANALYSIS_PARAMS_FILENAME: str = "BirdNET_analysis_params.csv"
LABEL_LANGUAGE: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US

#####################
# Training settings #
#####################

# Sample crop mode
SAMPLE_CROP_MODES = Literal["center", "first", "segments", "smart"]

# List of non-event classes
NON_EVENT_CLASSES: list[str] = ["noise", "other", "background", "silence"]

# Upsampling settings
UPSAMPLING_MODES = Literal["repeat", "mean", "smote"]

# Model output format
TRAINED_MODEL_OUTPUT_FORMATS = Literal["tflite", "raven", "both"]

# Model save mode (replace or append new classifier)
TRAINED_MODEL_SAVE_MODES = Literal["replace", "append"]

################
# Runtime vars #
################

ERROR_LOG_FILE: str = os.path.join(SCRIPT_DIR, "error_log.txt")
