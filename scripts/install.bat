REM Copyright (c) Microsoft Corporation.
REM Licensed under the MIT license.

REM Below will be required for Python 3.7 and below if pickle wasn't updated
REM otherwise ray.init() fails if below is not done
REM CALL conda install -y -c conda-forge pickle5

CALL conda install -y -c anaconda pydot graphviz
CALL pip install -e .[dev]