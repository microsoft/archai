#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Builds a docker image with standard arguments
docker build . --file Dockerfile --tag nvidia22.07-archai0.6.6:latest