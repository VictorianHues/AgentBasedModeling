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

.. code-block:: console

   $ make -j


To reproduce our results exactly as they are in the report, use the high-quality option.

.. code-block:: console

   $ QUALITY=high make -j


The generated results can be found in the created ``results`` directory.


