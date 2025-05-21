# Contributing

👍🎉 First off, thank you for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## What Should I Know Before I Get Started?

### How Do I Start Contributing?

The below workflow is designed to help you begin your first contribution journey. It will guide you through creating and picking up issues, working through them, having your work reviewed, and then merging.

Help on open source projects is always welcome and there is always something that can be improved. For example, documentation (like the text you are reading now) can always use improvement, code can always be clarified, variables or functions can always be renamed or commented on, and there is always a need for more test coverage. If you see something that you think should be fixed, take ownership! Here is how you get started:

## How Can I Contribute?

For any contributions that need design changes/API changes, reach out to maintainers to check if an Architectural Design Record would be beneficial. Reason for ADR: teams agree on the design, to avoid back and forth after writing code. An ADR gives context on the code being written. If requested for an ADR, make a contribution [using the template](./architecture_records/template.md).

When contributing, it's useful to start by looking at [issues](https://github.com/IBM/deepview/issues). After picking up an issue, writing code, or updating a document, make a pull request and your work will be reviewed and merged. If you're adding a new feature or find a bug, it's best to [write an issue](https://github.com/IBM/deepview/issues/new/choose) first to discuss it with maintainers. 

When your contribution is ready, you can create a pull request. Pull requests are often referred to as "PR". In general, we follow the standard [GitHub pull request](https://help.github.com/en/articles/about-pull-requests) process. Follow the template to provide details about your pull request to the maintainers. It's best to break your contribution into smaller PRs with incremental changes, and include a good description of the changes. We require new unit tests to be contributed with any new functionality added. 

Before sending pull requests, make sure your changes pass formatting, linting and unit tests. These checks will run with the pull request builds. Alternatively, you can run the checks manually on your local machine [as specified below](#development).

#### Dependencies
If additional new Python module dependencies are required, think about where to put them:

- If they're required for `deepview`, then append them to the [dependencies](...) in the `pyproject.toml`.
- If they're optional dependencies for additional functionality, then add them in the `pyproject.toml` file like were done for ...
- If it's an additional dependency for development, then add it to the [dev](...) dependencies.

#### Code Review

Once you've [created a pull request](#how-can-i-contribute), maintainers will review your code and may make suggestions to fix before merging. It will be easier for your pull request to receive reviews if you consider the criteria the reviewers follow while working. Remember to:

- Run tests locally and ensure they pass
- Follow the project coding conventions
- Write detailed commit messages
- Break large changes into a logical series of smaller patches, which are easy to understand individually and combine to solve a broader issue

Maintainers will perform "squash and merge" actions on PRs in this repo, so it doesn't matter how many commits your PR has, as they will end up being a single commit after merging.

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report ✏️, reproduce the behavior 💻, and find related reports 🔎.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues using the Bug Report template](https://github.com/IBM/deepview/issues/new?template=bug_report.md). Create an issue on that and provide the information suggested in the bug report issue template. 

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features, tools, and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion ✏️ and find related suggestions 🔎

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues using the Feature Request template](https://github.com/IBM/deepview/issues/new?template=feature_request.md). Create an issue and provide the information suggested in the feature requests or user story issue template.

#### How Do I Submit A (Good) Improvement Item?

Improvements to existing functionality are tracked as [GitHub issues using the User Story template](https://github.com/IBM/deepview/issues/new?template=user_story.md). Create an issue and provide the information suggested in the feature requests or user story issue template.

## Development

### Set up your dev environment

The following tools are required:

- [git](https://git-scm.com)
- [python](https://www.python.org) (v3.11)
- [pip](https://pypi.org/project/pip/) (v23.0+)

You can setup your dev environment using `tox`, an environment orchestrator which allows for setting up environments for and invoking builds, unit tests, formatting, linting, etc. Install `tox` with:

```shell
pip install tox
```

If you want to manage your own virtual environment instead of using `tox`, you can install deepview and all dependencies. Check out [installation](./README.md#installation) for more details.

Before pushing changes to GitHub, you need to run the tests, coding style and spelling check as shown below. They can be run individually as shown in each sub-section or can be run with the one command:

```shell
tox
```

### Unit tests

Unit tests are enforced by the CI system. When making changes, run the tests before pushing the changes to avoid CI issues. Running unit tests ensures your contributions do not break exiting code. We use [pytest](https://docs.pytest.org/) framework to run unit tests. The framework is setup to run all run all test_*.py or *_test.py in the [tests](./tests) directory.

Running unit tests is as simple as:

```sh
tox -e unit
```

By default, all tests found within the tests directory are run. However, specific unit tests can run by passing filenames, classes and/or methods to `pytest` using `tox` positional arguments. The following example invokes a single test method `text_x` that is declared in the `tests/sub_dir/test_filename.py` file:

```shell
tox -e unit -- tests/subdir/test_filename.py::text_x
```

### Coding style

Deepview follows the Python [pep8](https://peps.python.org/pep-0008/) coding style. The coding style is enforced by the CI system, and your PR will fail until the style has been applied correctly.

We use [Ruff](https://docs.astral.sh/ruff/) to enforce coding style using [Black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/), and [Flake8](https://docs.astral.sh/ruff/faq/#how-does-ruffs-linter-compare-to-flake8).

You can invoke Ruff with:

```sh
tox -e ruff
```

You could optionally install the git [pre-commit hooks](https://pre-commit.com/) if you would like to format the code automatically for each commit:

```shell
pip install pre-commit
pre-commit install
```

In addition, we use [pylint](https://www.pylint.org/) to perform static code analysis of the code.

You can invoke the linting with the following command

```shell
tox -e lint
```

### Spelling check

Spelling check is enforced by the CI system. Run the checker before pushing the changes to avoid CI issues. We use [pyspelling](https://github.com/facelessuser/pyspelling) spell check automation tool. It is a wrapper around CLI of [Aspell](http://aspell.net/) and [Hunspell](https://hunspell.github.io) which are spell checker tools. We configure `pyspelling` to use `Aspell` as the spell checker tool of choice.

Running the spelling check is as simple as:

```sh
tox -e spellcheck
```

## Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issues:

- Issues with the [`good first issue` label](https://github.com/IBM/deepview/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - these should only require a few lines of code and are good targets if you're just starting contributing.
- Issues with the [`help wanted` label](https://github.com/IBM/deepview/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) - these range from simple to more complex, but are generally things we want but can't get to in a short time frame.