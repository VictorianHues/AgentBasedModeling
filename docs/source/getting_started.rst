Getting started
===============

Setting up
----------

First things first, clone the project repository from GitHub using the GitHub CLI:

.. code-block:: console

   $ gh repo clone VictorianHues/AgentBasedModeling

You can also clone the repository using HTTPS or SSH.

This project uses `uv <https://docs.astral.sh/uv/>`_ for Python environment management.
To ensure consistent results, we recommend using uv when working with this project or 
re-running experiments. The following steps assume that you are using uv.

To reproduce our results, first configure the Python environment and install all 
required dependencies:

.. code-block:: console

   $ uv sync

And then run your simulation code:

.. code-block:: console

   $ make

For a parallel run, use:

.. code-block:: console

   $ make -j

To inspect available experiment and plotting workflows, use:

.. code-block:: console

   $ make help

.. note::

   The default ``make``/``make -j`` pipeline now includes the CCDF gamma sweep
   at 0.0041, 0.0045, and 0.0049.

.. important::

   If you are trying to recreate the results from the `report 
   <https://github.com/VictorianHues/AgentBasedModeling/blob/main/project_report.pdf>`_, 
   visit the :doc:`experiments` section for details on how to run the experiments. 

What next? See :doc:`contributing` to contribute changes to the project.
