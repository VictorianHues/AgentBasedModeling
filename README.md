# Agent-based Modeling Project

Victoria Peterson - 15476758

Karolina Chłopicka - 15716546

Henry Zwart - 15393879

Shania Sinha - 14379031

## Table of Contents

1. [Overview](#overview)
2. [Usage and Installation](#usage-and-installation)
3. [Implementation](#implementation)
4. [Contributing](CONTRIBUTING.md)
5. [License](#license)

## Overview

## Usage and Installation

To get started, first clone this repository:

```sh
gh repo clone VictorianHues/AgentBasedModeling
```

This project uses [uv](https://docs.astral.sh/uv/) for Python package environment and
environment reasons. To ensure consistent results we recommend using uv when 
running experiments. It is also possible to run experiments using 
[venv](https://docs.python.org/3/library/venv.html), though care must be taken to set up
the Python environment correctly.


### Usage (uv)

1. Configure Python environment and install any required dependencies:

    ```sh
    uv sync
    ```

2. Run simulations:

    ```sh
    uv run scripts/my_script.py
    ```

### Usage (venv)

> [!IMPORTANT]
> These instructions assume that you have a correct version of Python available 
> (see version specifier in [pyproject.toml](pyproject.toml)).
> In the following instructions we refer to the Python executable as `python`, 
> however, this may differ on your machine.

1. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install Dependencies**: Ensure you have all the required dependencies installed. You can install them using the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

3. **Initialize Directories as Packages**: This step will ensure that all function imports from different directories are recognized by initializing the directories as packages in the virtual environment. This step may be unnecessary if you established the environmental variable `PYTHON_PATH` for your imports:

    ```sh
    pip install -e .
    ```

4. **Run Simulations**: 

5. **Generate Plots**: 

## Implementation

### Source Code

### Scripts

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
