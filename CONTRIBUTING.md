# Contributing

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package environment and
environment reasons. Even if you are not modifying the project dependencies, we advise
using uv to ensure consistent runtime and development environments.

We also use [pre-commit](https://pre-commit.com/) for automatic code-formatting prior 
to git commits. Install pre-commit and initialise the hooks as follows:

```sh
# With uv.
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# With pip (ensure you are in a virtual environment first).
pip install pre-commit

# Check it installed okay
pre-commit --version  # Should say 'pre-commit 4.2.0' or similar

# Initialise the pre-commit hooks for this repository
pre-commit install

# Finally, its a good idea to test that it works as intended
pre-commit run --all-files
```

If the above steps worked as expected, pre-commit should now automatically 
lint and format any Python code prior to git commits, preventing accidentally
checking in problematic code.

## Contribution guide

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Ensure all tests pass, with `uv run pytest`
5. Push your changes to your forked repository.
6. Create a pull request to the main repository.


