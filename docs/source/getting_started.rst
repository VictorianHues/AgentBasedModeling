Getting started
===============

Setting up
----------

First things first, clone the project repository from GitHub:

.. code-block:: console

   $ gh repo clone VictorianHues/AgentBasedModeling

This project uses `uv <https://docs.astral.sh/uv/>`_ for Python environment management.
To ensure consistent results, we recommend using uv when working with this project or 
re-running experiments. The following steps assume that you are using uv.

To reproduce our results, first confgure the Python environment and install all 
required dependencies:

.. code-block:: console

   $ uv sync

And then run the simulation code:

.. code-block:: console

   $ uv run scripts/my_script.py


What next? See :doc:`contributing` to contribute changes to the project, or :doc:`experiments`
to reproduce the experimental results in the corresponding report.



