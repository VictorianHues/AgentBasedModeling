.. ABM Project documentation master file, created by
   sphinx-quickstart on Tue Jun 10 12:37:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Climate-related Decision-making ABM
===================================

This project uses agent-based modelling to investigate the feedback cycle between 
public support for climate action and individually-perceived climate change severity.
We extend existing work in this area to explore how system dynamics and equilibria 
differ in the coupled model when agents are heterogeneous in their preferences and 
perception of the environment. You can find the associated report 
:download:`here <../../project_report.pdf>`.

.. image:: figures/11b_env.gif
   :width: 70%
   :align: center

.. image:: figures/phase_portraits.png
   :width: 70%
   :align: center

All code is written in Python 3.13. The core model, 
:py:class:`abm_project.vectorised_model.VectorisedModel`, is implemented as a Python
class wrapping efficient vectorised (Numpy) data structures. The model is designed 
with extensibility in mind, with key components modularised and configurable at 
model initialisation. For instance, it is straightforward to implement new methods 
for updating agents' local environments, which are passed to the model as functions. 

The repository itself is structured as a Python package which contains reusable 
components. Specific experiments are separated out into scripts, which are orchestrated
by a `Makefile <https://github.com/VictorianHues/AgentBasedModeling/blob/main/Makefile>`_.
We use `uv <https://docs.astral.sh/uv/>`_ to ensure a consistent, version-locked Python 
environment for experimentation.

Core modules in the Python package (``abm_project``) include:

* :ref:`abm_project.vectorised_model <vectorised-model-api>`: The core model, used in all of 
  our experiments.

* :ref:`abm_project.mean_field <mean-field-api>`: A mean-field approximation to the core model. 
  Includes functions to solve for equilibria, expected mean action, and running simulations using the derived mean-field dynamical system. 

* :ref:`abm_project.oop_model <oop-model-api>`: A predecessor to the ``VectorisedModel``, whose 
  implementation prioritises ease-of-use and descriptivity. **Note:** the ``OOPModel`` is not 
  actively maintained, and does not reflect the current state of ``VectorisedModel``.

* :ref:`abm_project.kraan <kraan-api>`: An implementation of the 2D lattice model described in 
  `Kraan et al <https://www.sciencedirect.com/science/article/pii/S2214629618301932>`_, which 
  inspired our work.

Contents
--------

Check out the :doc:`getting_started` section for details on how to contribute to this
project, or :doc:`experiments` to reproduce our experiments.

.. toctree::
   :maxdepth: 1

   getting_started
   experiments
   api
   contributing



The repository is licensed under the `MIT License <https://github.com/VictorianHues/AgentBasedModeling/blob/main/LICENSE>`_. 


