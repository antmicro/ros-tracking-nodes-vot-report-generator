# Video Object Tracking report generator

Copyright (c) 2020-2021 [Antmicro](https://www.antmicro.com)

*Check how this repository can be used in* [ros-tracking-policy-examples repository](https://github.com/antmicro/ros-tracking-nodes-policy-examples).

A python tool for generating reports regarding detection and tracking of a single object in video sequences.

## Usage

To generate reports run `main.py` script. It takes exactly two positonal arguments: `output_path` and `input_path`, where:
* `output_path` is path to directory with test results; format of this repository is specified in the [ros-tracking-nodes-tester-node repository](https://github.com/antmicro/ros-tracking-nodes-tester-node).
* `input_path` path to directory containing dataset which was used to run the tests; the format is also described in the above-mentioned repository.

## Generated reports

Reports are generated as html files and `.png` images.
