# Minimal reproducible scripts for known failures (unit tests)

> Testing framework for Deepview. Here we have unit tests that ensure proper functionality of the different Deepview features and e2e tests for different models with known cases for unsupported operations and layer debugging functionalities.

## Folder structure

The code is structured in a way following the Deepview source code directory with one test file for each one of the files from the source code directory of Deepview:

* `models`: e2e tests for different models to test tooling and e2e behaviour of the debugging process.
    * `test_bamba9b_hf.py`: e2e tests for ibm-ai-platform/Bamba-9B-v1 from HuggingFace using transformers library that runs the model and then check for the unsupported compute operations.
    * `test_granite2b_fms.py`: e2e tests for ibm-granite/granite-3.3-2b-instruct ported to Foundation Model Stack that runs the model and then debug model layers to see if there is any errored layers running on the AIU systems.
    * `test_mistral7b_fms.py`: e2e tests for mistralai/Mistral-7B-Instruct-v0.3 ported to Foundation Model Stack that runs the model and then debug model layers and unsupported compute operations on the AIU systems.

## Installation

For the tests you need to have the following dependencies installed in our python env:

* `pytest`
* `pytest-cov`

Those can be installed with `pip3 install` or `pip3 install .[dev]` as is defined in the `pyproject.toml`.

## How to run the e2e tests

> To run the e2e tests as the state of current Deepview implementation to not have problem with subprocess leaking output from one model test case to another we need to run them sequentially. To do so please run each of the tests once the previous one ends:

```bash
pytest tests/models/<path_to_your_test_file>
```

To run our current e2e tests run it sequentially in the following order:

```bash
pytest tests/models/test_bamba9b_hf.py
```

```bash
pytest tests/models/test_granite2b_fms.py
```

```bash
pytest tests/models/test_mistral7b_fms.py
```

## Check for test coverage

We can check our test code coverage using the `pytest-cov` utility as follows:

```bash
pytest --cov=deepview tests/
```
