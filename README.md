# Restricted-Py-LaCAM: Enhanced Multi-Agent Pathfinding with Rotational Constraints

This repository is an enhanced version of the original py-lacam project, focusing on introducing preprocessing techniques for grid layer extraction and directional constraints. These enhancements aim to improve pathfinding performance in multi-agent systems by introducing rotational movement restrictions.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running Preprocessing](#running-preprocessing)
  - [Running Multiple Experiments](#running-multiple-experiments)
  - [Running a Single Experiment](#running-a-single-experiment)
- [Original py-lacam Instructions](#original-py-lacam-instructions)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project builds upon the [py-lacam repository](https://github.com/Kei18/py-lacam/tree/pibt) and introduces new preprocessing logic to improve multi-agent pathfinding by restricting agent movements based on grid layers and rotational constraints. The primary contributions include:
- **Preprocessing for Layer-Based Movement:** A method to segment grids into layers with defined movement directions (clockwise/counterclockwise).
- **Experiment Automation:** Scripts to run and analyze multiple experiments across different grid and agent configurations.
- **Visualization Tools:** Tools for visualizing the grid layers, direction assignments, and experiment results.


## Key Features

- **Preprocessing and Visualization:** Extracts concentric layers from grids, assigns movement directions, and visualizes the output.
- **Experiment Automation:** Allows for running multiple experiments with different configurations and saving the results in CSV format.
- **Flexible Experiment Setup:** Run experiments with or without preprocessing, across various grid maps and agent counts.

## Setup and Installation

### Prerequisites

Ensure you have Python 3.9+ and `pip` installed. The project dependencies can be installed using:
```
pip install -r requirements.txt
```

Alternatively, you can use poetry for managing dependencies:
```
poetry install
```

## Usage
### Running Preprocessing
The preprocess.py script preprocesses a given map, extracting layers, assigning directions, and visualizing the output: ```python run_experiments.py```

### Running a Single Experiment
The app.py script runs a single experiment based on the provided arguments. Here are two ways to run it:
1. Basic command: ```python app.py```
2. More detailed command with additional arguments: ```poetry run python app.py -m <path_to_map_file> -i <path_to_scenario_file> -N <number_of_agents> --no-flg_star```


