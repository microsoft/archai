Documentation
=============

The Archai project welcomes contributions through the implementation of documentation files using Sphinx and RST. If you are interested in contributing to the project in this way, please follow these steps:

#. Ensure that Sphinx is installed. You can install it using ``pip install archai[docs]``.

#. Check out the Archai codebase and create a new branch for your changes. This will allow for easy submission of your code as a pull request upon completion.

#. Create an ``.rst`` file in the :github:`docs`. For example, if writing an API documentation for the :github:`archai/trainers/nlp/hf_training_args.py` file, the corresponding path would be :github:`docs/reference/api/archai.trainers.nlp.rst`.

#. Check the pre-defined format and include the corresponding section:

    .. code-block:: rst

        Training Arguments
        ------------------

        .. automodule:: archai.trainers.nlp.hf_training_args
        :members:
        :undoc-members:

#. To build the documentation, run the following command from the root directory of the documentation (i.e. the ``docs`` directory). The HTML files will be created in a ``_build`` directory.

    .. tab:: Linux/MacOS

        .. code-block:: sh

            cd archai/docs
            make html

    .. tab:: Windows

        .. code-block:: bat

            cd archai/docs
            .\make.bat html

These are just some basic guidelines for an API documentation using Sphinx. For more information check the `Sphinx documentation <https://www.sphinx-doc.org/en/master>`_ and `RST guide <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_. Additionally, it is recommended to review the Archai documentation style guide before contributing to the documentation to ensure consistency.