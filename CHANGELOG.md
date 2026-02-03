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

## [1.0.0] - 2026/02/03

### Added 
* Added setup_model_handler wrapper function to reduce redundant code (\#155)
* Added TOX to pyproject.toml for development testing (\#166) 
* Added support for sorting layers by complexity and executing them sequential in layer-level debug mode. (\#167) 
* Added last_n_tokens parameter when using FMS model forward (\#178) 
* Added a utility script for automating DeepView testing on a list of models for a single mode of DeepView (\#180) 
* Added example scripts for using DeepView unsupported ops mode as a library (\#168)

### Changed
* Move model runner methods for each individual DeepView mode into its respective file in the core component (\#155)
* Update COMPILATION_MODE environment variable (\#161)
* Update lazy handle import and torch_sendnn convertor methods as they were renamed (\#164 ,\#172) 
* Update layer_io data structure in layer debug mode (\#167) 
* Update image in sample pod yaml (\#172) 
* Update pod yaml to use spyre scheduler and resources instead of aiu (\#183) 
* Refactor model handler to modular sub-handlers for scalability to new model types (\#156) 
* Move sample pod yaml to top-level directory(\#168) 

### Fixed
* Fix triton warning (#162) 
* Fix handling of kwargs and dynamo error in layer debug mode (\#167) 
* Fix lazy handle structure parsing in unsupported ops mode (\#176) 
* Limit Triton version to match PyTorch version (\#181)

## [0.2.0] - 2025/08/07

### Added

* Project restructuring and code updates to conform DeepView as an installable tool. (\#52)
* Added Threshold Testing functionality through AIU-FMS to extend the Layer compilation to test decoder model comparisions. (\#64)
* Added doc strings to all existing code base to improve code and API documenation. (\#69)
* Added Travis CI build and Copyright headers. (\#74 and \#78)
* Added separate log file for each DeepView mode. (\#80)
* Initial but limited support for HF models in layer debugging mode. (\#81)
* Migrate to Spyre DD2 infrastructure. This included several changes to the ENV variables, pod yaml, configurations, etc. (\#84)
* Added the needed dependencies for Mistral models. (\#90)
* Added Layerwise diversion as a new debugging mode. (\#101, \#103, )
* Graceful exit of HF model type in modes other than unsupported. (\#112)
* Combined AIU capture mode with AIU diversion mode to simplify the user experience for the new mode (Layer-diversion). (\#111)
* Added micro granite model for testing. (\#141)


### Changed
* Replaced the sendnn_decoder with sendnn backend as the former is deprecated. (\#68)
* Changed the layer debugging implementation to use the sendenn backend. (\#73) 
* Updated the pytest to integrate all the new additions to DeepView. (\#121)
* Updated the documentation for the examples. (\#123)
* Changed to pytest to cover the new layer-diversio mode and new results from Bamba. (\#133)
* Split Mistral pytests into two seperate files for each mode. (\#144)

### Fixed

* Fixed the python version requirement and removed MIT license. (\#55)
* Fixed file path depedencies for layer debugging and generation of repro code. (\#67)
* Fixed all formatting and linting issues. (\#65)
* Fixed FMS version and HF environment variables. (\#89)
* Fixed a bug fir Unsupported Ops mode. The arguments were passed in the wrong order. (\#119)
* Cleanup of the pod yaml file and fixes to the HF ENV variable to allow correct use of the HF cache. (\#120)
* Fixed bugs introduced with the latest AIU e2e image with layer debugging mode. (\#129)
* Fixed the repro-code output generation by introducing the right AIU ENV variables. (\#134)
* Fixed the error generated while trying to run base_model layer of fms models in layer_debugging mode. (\#135)
* Fixed mpnet pytest. (\#136)
* Fixes for running LLama and readme cleanup. (\#140)
* Fixed test assertions. (\#141)
* Added graceful exits for unsupported models and Readme updates. (\#143)


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
