# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]:

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.
## Unreleased

## [0.1.0] - 2025/05/20

### Added

* Initial public release.
* Added unsupported ops identification with detailed metadata (such as input shapes and dtypes) for FMS models. (\#3)
* Added packaging artifacts and ReadMe. (\#12 and \#14)
* Added repro code generation for unsupported ops. (\#17)
* Added support for running unsupported_op mode on HF models. (\#24)
* Added layer-wise debugging. (\#25 and \#28)
* Added HF support for layer debugging and separated out model handler functions. (\#30)
* Added sample pod yaml configuration. (\#42)
* Added documentation for model testing. (\#50)


[UNRELEASED]: https://github.com/IBM/deepview/compare/v0.2.0...HEAD
[0.1.0]: https://github.com/IBM/deepview/releases/tag/v0.1.0

[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
