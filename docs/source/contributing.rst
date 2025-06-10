Contributing
=============

You want to contribute to this project? Awesome! There are just a couple of 
housekeeping things to sort out before you dive in.

uv
---

`uv <https://docs.astral.sh/uv/>`_ is a Python environment and package management tool.
We use it in this repository to keep everything consistent and reproducible. If you don't
already have uv installed, now is the time to do so :^)

Once this is installed, configure the Python environment and download all required 
dependencies by navigating to this repository and running:

.. code-block:: console

   $ uv sync

This creates a virtual environment under ``.venv``, which is used implicitly whenever
you run a uv command. *You do not need to manually activate this virtual
environment*.

Python packages can be added or removed from the project using ``uv add <PACKAGE_NAME>``
and ``uv remove <PACKAGE_NAME>``. Doing so will update both:

#. The pyproject.toml file, which maintains a human-readable project configuration, and
#. The uv.lock file, which is a machine-generated file used to specify the project dependency
   graph. 

Note that the uv.lock file is not intended to be modified manually (or even really looked at). 
For the most part, you can forget it exists.

pre-commit and ruff
--------------------

We also use `pre-commit <https://pre-commit.com/>`_ with a 
`ruff <https://docs.astral.sh/ruff/>`_ hook to automatically format all Python
code prior to git commits.

The specific ruff rules which will be applied are specified in the repository 
pyproject.toml file. For more information, and a description of these rules, see
the `ruff documentation <https://docs.astral.sh/ruff/rules/>`_.

Installing pre-commit is straightforward using ``uv tool``:

.. code-block:: console

   $ uv tool install pre-commit --with pre-commit-uv --force-reinstall

   # Check installation was successful
   $ pre-commit --version  # Should say 'pre-commit 4.2.0' or similar

Next you need to initialise pre-commit for this specific repository. This
step will install the pre-commit hooks which will be run prior to every git
commit:

.. code-block:: console

   # Initialise the pre-commit hooks
   $ pre-commit install

   # Check that everything works as intended (this should succeed)
   $ pre-commit run --all-files

If the above steps worked as expected, pre-commit should now automatically 
lint and format any Python code prior to git commits, preventing accidentally
checking in problematic code.

Project documentation
----------------------

Finally, we use sphinx to build the project documentation. This can also be installed
with uv as follows:

.. code-block:: console

   $ uv tool install sphinx --force-reinstall

   # Check installation was successful
   $ sphinx-build --version  # Should say 'pre-commit 8.2.3' or similar

The documentation is automatically rebuilt whenever new changes are push to the main 
branch. However, its never a bad idea to check that everything still works locally 
after making changes. 

To build the documentation locally:

.. code-block:: console

   $ cd docs && make clean html && cd ..

Open the freshly-built documentation with:

.. code-block:: console

   $ open docs/build/html/index.html

If everything worked correctly, the documentation should open in your browser.

Adding new content
^^^^^^^^^^^^^^^^^^^

Sphinx will automatically add any newly-documented functions in existing modules.
If you are adding a new module, you will need to add it to the ``docs/source/api.rst`` file.
There are already a few examples in there.


Contribution process
----------------------

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Ensure all tests pass, with `uv run pytest`
5. Push your changes to your forked repository.
6. Create a pull request to the main repository.
7. If all GitHub CI checks are successful, great! It's time to request a review 
   from one of the project maintainers.


