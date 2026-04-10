Reproducing experiments
=======================

The experiments for the corresponding report are orchestrated using 
`make <https://www.gnu.org/software/make/>`_, so reproducing our results
is a breeze.

If you haven't already done so, see the :doc:`getting_started` section to 
configure a Python environment with the expected packages and package versions.

We provide two "quality" modes for running the experiments: low (the default) and high. 
The former runs computationally expensive experiments with reduced repeats, parameters, 
or iterations. This is useful for getting things running quickly, but the results
won't reflect those present in the report. 

Run the full default pipeline (figures and dependencies) with:

.. code-block:: console

   $ make -j

This includes the CCDF gamma sweep at 0.0041, 0.0045, and 0.0049.

To reproduce our results exactly as they are in the report, use the high-quality option.

.. code-block:: console

   $ QUALITY=high make -j

Run only data-generation targets (no plotting-only reruns):

.. code-block:: console

   $ make measure-all
   $ make measure-all QUALITY=high

Run only plotting targets from existing data:

.. code-block:: console

   $ make plot-all
   $ make plot-all QUALITY=high

Phase-transition workflows:

.. code-block:: console

   $ make phase-transition-analysis
   $ make phase-transition-all-configurations

CCDF gamma sweep only:

.. code-block:: console

   $ make powerlaw-spatial-corr-gamma-sweep

Worker/process tuning for heavy simulation runs:

.. code-block:: console

   $ make -j ABM_MAX_WORKERS=4 ABM_CHUNKSIZE=2

Inspect available targets and expanded commands:

.. code-block:: console

   $ make help
   $ make -n phase-transition-analysis


Generated output is written to ``data/`` (intermediate measurements),
``results/figures/`` (plots), and ``results/animations/``.


