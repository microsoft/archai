Installation
============

There are various methods to install Archai, but it is recommended to use it within a virtual environment, such as ``conda`` or ``pyenv``. This ensures that the software runs in a consistent and isolated environment, and allows for easy management of installed packages and dependencies.

.. attention::

   Archai requires `Python <http://python.org>`_ 3.7+ and `PyTorch <https://pytorch.org>`_ 1.7.0+.

PyPI
----

PyPI provides a convenient way to install Python packages, as it allows users to easily search for and download packages, as well as automatically handle dependencies and other installation requirements. This is especially useful for larger Python projects that require multiple packages to be installed and managed.

.. code-block:: sh

    pip install archai

The default installation only includes the core functionality of Archai, e.g., NAS-related packages. To install additional functionalities, use the following commands:

* Computer Vision: ``pip install archai[cv]``.
* Natural Language Processing: ``pip install archai[nlp]``.
* DeepSpeed: ``pip install archai[deepspeed]``.
* Flash-Attention: ``pip install archai[flash-attn]``.
* Documentation and Notebooks: ``pip install archai[docs]``.
* Unit Tests: ``pip install archai[tests]``.
* Development: ``pip install archai[dev]``.

Docker
------

Docker is a useful tool for running experiments because it provides a consistent, isolated environment for the experiment to run in. This ensures that the results of the experiment are not affected by external factors, such as the specific operating system or installed packages on the host machine.

The :github:`docker/Dockerfile` provides a development environment to run experiments. Additionally, :github:`docker/build_image.sh` and :github:`docker/run_container.sh` provide scripts to build the image and run the container, respectively:

.. code-block:: sh

    cd docker
    bash build_image.sh
    bash run_container.sh
