Unitary Tests
=============

The Archai project welcomes contributions through the implementation of unitary tests using Pytest. If you are interested in contributing to the project in this way, please follow these steps:

#. Ensure that Pytest is installed. You can install it using ``pip install archai[tests]``.

#. Check out the Archai codebase and create a new branch for your changes. This will allow for easy submission of your code as a pull request upon completion.

#. Create a ``.py`` test file in the :github:`tests` directory. For example, if writing a unitary test for the :github:`archai/nlp/trainers/hf/training_args.py` file, the corresponding path would be :github:`tests/nlp/trainers/hf/test_hf_training_args.py`.

#. Write test functions inside the created file. These functions should be named using the pattern: ``test_<name of function being tested>``.

#. Test functions should utilize the assert statement to verify that the output of the tested function is correct. For example:

    .. code-block:: python

        from archai.nlp.trainers.hf.training_args import DistillerTrainingArguments

        def test_distiller_training_arguments():
            args = DistillerTrainingArguments("tmp")
            assert args.alpha == 0.5
            assert args.temperature == 1.0

            args = DistillerTrainingArguments("tmp", alpha=0.75, temperature=1.25)
            assert args.alpha == 0.75
            assert args.temperature == 1.25

#. Run your tests using the ``pytest`` command. This will automatically discover and execute all test functions in the tests directory.

#. To run a specific test file, use the ``pytest`` command followed by the path to the desired file. For example: ``pytest tests/eval/profiler/test_profiler_utils.py``.

After writing and running your tests, you may submit your code as a pull request to the main project repository. Please include a description of the changes made and any relevant issue references.