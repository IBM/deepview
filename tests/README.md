# Minimal reproducible scripts for known failures (unit tests)

> Testing framework for Deepview. Here we have unit tests that ensure proper functionality of the different Deepview features and e2e tests for different models with known cases for unsupported operations and layer debugging functionalities.

## Folder structure

The code is structured in a way following the Deepview source code directory with one test file for each one of the files from the source code directory of Deepview:

* `core`: unit tests for core functionality of Deepview.
    * `conftest.py`: setup configuration to run a model debug on each test based on micro-models to test Deepview functionality.
    * `test_hook_monitor.py`: tests for the hooks monitoring system.
    * `test_model_runner.py`: tests to check the run model functionality, main behaviour to load, run and start the process of debug models with Deepview.
    * `test_unsupported_ops.py`: tests for the funtionality that analyzies and retrieves the compute operations that are still not supported by the AIU systems.
* `models`: e2e tests for different models to test tooling and e2e behaviour of the debugging process.
    * `test_bamba9b_hf.py`: e2e tests for ibm-ai-platform/Bamba-9B-v1 from HuggingFace using transformers library that runs the model and then check for the unsupported compute operations.
    * `test_granite2b_fms.py`: e2e tests for ibm-granite/granite-3.3-2b-instruct ported to Foundation Model Stack that runs the model and then debug model layers to see if there is any errored layers running on the AIU systems.
    * `test_mistral7b_fms.py`: e2e tests for mistralai/Mistral-7B-Instruct-v0.3 ported to Foundation Model Stack that runs the model and then debug model layers and unsupported compute operations on the AIU systems.

## Installation

For the tests you need to have the following dependencies installed in our python env:

* `pytest`
* `pytest-cov`

Those can be installed with `pip3 install` or `pip3 install .[dev]` as is defined in the `pyproject.toml`.

## Run e2e tests

> To run the e2e tests as the state of current Deepview implementation to not have problem with subprocess leaking output from one model test case to another we need to run them sequentially. To do so please run each of the tests once the previous one ends:

```bash
pytest tests/models/<path_to_your_test_file>
```

Example:

```bash
pytest tests/models/test_mistral7b_fms.py
```

## How to run the tests

To run the tests you can simply run:

```bash
pytest
```

This should pick up all the tests defined in `tests` folder in the root of the project.

If we want to run a particular test file:

```bash
pytest <path_to_your_test_file>
```

For example if we want to run the unsupported_ops tests:
```bash
pytest tests/core/test_unsupported_ops.py
```

Same happens if we want to run a unique test:

```bash
pytest <path_to_test_file>::<name_of_the_test>
```

For example if we want to run the unsupported_ops test with no unsupported ops:
```bash
pytest tests/core/test_unsupported_ops.py::test_get_unsupported_ops_no_unsupported
```

## Check for test coverage

We can check our test code coverage using the `pytest-cov` utility as follows:

```bash
pytest --cov=deepview tests/
```
