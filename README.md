# aiu-model-debugger
Diagnostic and debugging tools library to identify gaps in the `sendnn` compiler while the model is being enabled into low-level decomposable operations

# Initial directory structure 

aiu-model-debugger
├── LICENSE
├── README.md
├── configs
│   ├── README.md
│   └── debug_profiles.json
├── core
│   ├── README.md
│   ├── correctness.py
│   ├── fx_graph_analyzer.py
│   ├── hook_monitor.py
│   ├── model_runner.py
│   ├── op_mapper.py
│   └── unsupported_db.py
├── examples
│   └── README.md
├── scripts
│   ├── README.md
│   ├── isolate_layer.py
│   └── run_full_model.py
├── setup.py
├── tests
│   ├── README.md
│   └── test_cases
└── utils
    ├── README.md
    └── logger.py