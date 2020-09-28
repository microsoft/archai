
REM ray.init() fails if below is not done
CALL conda install -y -c conda-forge pickle5
CALL conda install -y -c anaconda pydot graphviz

pip install -e .