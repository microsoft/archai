# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re


def test_seed_patterns():
    pat = r"seed\_?([0-9]*[.])?[0-9]+"
    tests = [
        "dsas_3321_3.4_seed_232_sdasdsa234",
        "dsas_3321_3.4_seed232_sdasdsa234",
        "dsas_3321_3.4_seed_23.2_sdasdsa234",
        "dsas_3321_3.4_seed_232_sdasdsa234",
        "dsas_3321_3.4_seedseed232_sdasdsa234",
        "dsas_3321_3.4_seeed232_sdasdsa234",
    ]

    for test in tests:
        replaced = re.sub(pat, "", test)
        print(test, replaced)


test_seed_patterns()
