# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constants definitions used within the NLP package.
"""

import re

# Lazy loader-related
LIBRARY_PATH = 'archai.nlp.models'

# Models-related
BATCH_SIZE = 1
SEQ_LEN = 32

# ONNX-related
OMP_NUM_THREADS = 1
OMP_WAIT_POLICY = 'ACTIVE'

# Scoring metrics-related
WORD_TOKEN_SEPARATOR = "Ġ \nĊ\t\.;:,\'\"`<>\(\)\{\}\[\]\|\!@\#\$\%\^\&\*=\+\?/\\_\-~"
WORD_TOKEN_SEPARATOR_SET = set(WORD_TOKEN_SEPARATOR)
RE_SPLIT = re.compile('^(.*)([' + WORD_TOKEN_SEPARATOR + '].*)$', re.MULTILINE | re.DOTALL)
TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE = 65536
