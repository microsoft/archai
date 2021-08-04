REM requirement for Python 3.7 and below if pickle wasn't updated
REM otherwise ray.init() fails if line below is not done
REM CALL conda install -y -c conda-forge pickle5

CALL conda install -y -c anaconda pydot graphviz

pip install -e .