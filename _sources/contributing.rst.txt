Contributing
=============

You want to contribute to this project? Awesome! There are just a couple of 
housekeeping things to sort out before you dive in.

Already set up? Check out the :ref:`process for making contributions <contribution-process>`,
or :ref:`orchestrating new experiments <experiment-orchestration>`.

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

Finally, we use Sphinx to build the project documentation. The documentation is 
automatically rebuilt whenever new changes are push to the main branch. However, 
its never a bad idea to check that everything still works locally after making 
changes. 

To build the documentation locally:

.. code-block:: console

   # uv will handle the Sphinx dependencies for you
   $ make documentation

Open the freshly-built documentation with:

.. code-block:: console

   $ open docs/build/html/index.html

If everything worked correctly, the documentation should open in your browser.

Adding new content
^^^^^^^^^^^^^^^^^^^

Sphinx will automatically add any newly-documented functions in existing modules.
If you are adding a new module, you will need to add it to the ``docs/source/api.rst`` file.
There are already a few examples in there.


.. _contribution-process:

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


.. _experiment-orchestration:

Experiment orchestration
-------------------------

The experiments in this project are orchestrated using `Make <https://www.gnu.org/software/make/>`_
via a `Makefile <https://github.com/VictorianHues/AgentBasedModeling/blob/main/Makefile>`_ 
in the repository root. This process helps with both experimental reproducibility and 
computational efficiency. 

Orchestrating experiments in code allows users to reproduce our results 
without needing to comprehend a web of dependent scripts. Separating experiments into independent 
components increases both execution speed (via parallel execution) and development speed (by making
it straightforward to re-run only the necessary code after a change -- for instance, not re-running
analysis code when only the plotting functions have changed).

To reproduce all current results, simply run ``make`` from the project root. To run the pipeline 
components in parallel (where Make decides that this is possible), run ``make -j``.

Adding new experiments/results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use a repetitive pattern in the 
`Makefile <https://github.com/VictorianHues/AgentBasedModeling/blob/main/Makefile>`_, 
so it is typically straightforward to add new experiments. The general process is as 
follows:

1. Decide how many scripts you need.
""""""""""""""""""""""""""""""""""""

If the experiment comprises a computationally-heavy "simulation" or "analysis" 
section as well as a lightweight "plotting" section, we advise separating these into 
two or more scripts. When we do this, we set up a dependency chain in the Makefile. 
Make can then figure out exactly what needs to run whenever we make a change in the 
code. For instance, if we change the "simulation" script, it will run both this and 
the plotting script; if we only change the "plotting" script, it will only run this 
one. 

2. Create your scripts.
""""""""""""""""""""""""""""""""""""

Scripts for the Makefile are contained in the 
`scripts <https://github.com/VictorianHues/AgentBasedModeling/tree/main/scripts>`_ 
directory. When implementing multiple scripts for an experiment, it is good practice
to use the same name for them, with a descriptive prefix. For example:

* ``scripts/measure_equilibrium_environment_vs_rationality.py``

* ``scripts/plot_equilibrium_environment_vs_rationality.py``

Scripts should follow a general structure:

.. code-block:: python

   from pathlib import Path

   import some_module
   import another_module

   def main(savedir: Path, arg1, arg2, ...):
      ...

   if __name__ == "__main__":
      DATA_DIR = Path("data")
      FIGURES_DIR = Path("results/figures")

      PARAM_1 = ...
      PARAM_2 = ...
      PARAM_3 = ...

      main(FIGURES_DIR, PARAM_1, PARAM_2, PARAM_3)

The ``main`` function is responsible for saving interim data, figures, or other 
results. Any input data should be loaded prior to this and passed to ``main`` as
an argument.

.. important::

   Results should be saved in ``data/`` (for interim data), ``results/figures``, 
   or  ``results/animations``. These directories are created automatically by 
   Make. You do not need to create them manually. 

3. (Optional) add "fast-execution" parameters
"""""""""""""""""""""""""""""""""""""""""""""

The Makefile supports two operation modes: low quality (the default) and high quality.
The first allows experiments to be run under reduced settings (repeats, timesteps, 
etc.) for quick testing. If a script is computationally heavy, we recommend adding 
support for the low-quality mode. 

This is straightforward with the following pattern, which assumes the Makefile will 
run the script with a command-line argument specifying the mode (``quick == True``). 
Modified lines are annotated with a #!. Note that the figure is now saved with the 
filename indicating which mode we've generated it in.

.. code-block:: python

   import argparse  #!

   from pathlib import Path

   import some_module
   import another_module

   def main(savedir: Path, quality_label: str, arg1, arg2, ...):  #!
      ...

      fig.savefig(savedir / f"figure_name_{quality_label}_quality.pdf", bbox_inches="tight") #!

   if __name__ == "__main__":
      DATA_DIR = Path("data")
      FIGURES_DIR = Path("results/figures")

      PARAM_1 = ...
      PARAM_2 = ...
      QUICK_PARAM_3 = ... #!
      FULL_PARAM_3 = ... #!

      # === New code
      parser = argparse.ArgumentParser()
      parser.add_argument("--quick", action="store_true")
      args = parser.parse_args()

      if args.quick:
         quality_label = "low"
         param_3 = QUICK_PARAM_3
      else:
         quality_label = "high"
         param_3 = FULL_PARAM_3

      # === End of new code

      main(FIGURES_DIR, quality_label, PARAM_1, PARAM_2, param_3) #!


Now it's time to add our scripts to the Makefile.

4. Specifying what we want Make to create.
"""""""""""""""""""""""""""""""""""""""""""""

Lets suppose that our figure name is, ``my_figure.pdf``.

We need to let Make know that we want it to generate our figure. To do this, add 
the figure name to the ``FIGURE_NAMES`` variable near the top of the Makefile. 
Note that you'll need to add an escape character (``\``) at the end of the line, or 
the line before, to ensure that Make doesn't insert a linebreak.

5. Specifying how to create it.
"""""""""""""""""""""""""""""""""""""""""""""

Now suppose that our figure is created by ``scripts/plot_my_figure.py``, and that 
before this, we need to generate some data from a set of simulations with a separate
script: ``scripts/measurements_my_figure.py``.

When reading the Makefile, Make first looks at the "targets" to determine what 
it needs to create. In our case, these are the figure names, with ``$(FIGURES_DIR)``
appended to the start. Make then looks for a "rule" which generate this target.
Make rules look like this:

.. code-block:: make

   target: dependencies
      recipe

Where ``recipe`` is some code which generates the file with the same name as 
``target``.

In our example, there are three dependencies: the script which generates the 
figure, the input data, and the figures directory (this needs to be created 
before our script runs). Assuming we run our code with uv, our make rule will
be:

.. code-block:: make

   results/figures/my_figure.pdf: \             # Output file name
               scripts/plot_my_figure.py \      # First dependency
               data/my_figure_input_data.npz \  # Second dependency
               | results/figures                # Third dependency
      uv run scripts/plot_my_figure.py          # Run the script

This is a bit of a mouthful, but Make offers some shortcuts which simplify it a bit.
Firstly, our figures and data directories are saved as variables in the Makefile, so
we can reference them with ``$(FIGURES)`` and ``$(DATA)``. Second, ``uv run`` is saved 
as ``$(ENTRYPOINT)``. Third, the shorthand ``$<`` refers to our first dependency -- i.e.,
the script we want to run! 

So in simplified form, we have:

.. code-block:: make

   $(FIGURES_DIR)/my_figure.pdf: \                       # Output file name
               scripts/plot_my_figure.py \               # First dependency
               $(DATA_DIR)/my_figure_input_data.npz \    # Second dependency
               | $(FIGURES_DIR)                          # Third dependency
      $(ENTRYPOINT) $<                                   # Run the script


At this stage Make will complain, since we haven't told it how to find one of
our dependencies (the input data). We need to add a second make rule for this:

.. code-block:: make

   $(DATA_DIR)/my_figure_input_data.npz: \             # Output file name
               scripts/measurements_my_figure.py \     # First dependency
               | $(DATA_DIR)                           # Second dependency
      $(ENTRYPOINT) $<                                 # Run the script

And presto! Run ``make``, and your scripts should be added to the experiment 
orchestration.


