# -*- coding: iso-8859-1 -*-
# this file exists to mark this directory as a python module

R""" Hard particle Monte Carlo

HPMC performs hard particle Monte Carlo simulations of a variety of classes of shapes.

.. rubric:: Overview

HPMC implements hard particle Monte Carlo in HOOMD-blue.

.. rubric:: Logging

The following quantities are provided by the integrator for use in HOOMD-blue's :py:class:`hoomd.analyze.log`.

- ``mcm_sweep`` - Number of sweeps completed since the start of the MC integrator
- ``mcm_translate_acceptance`` - Fraction of translation moves accepted (averaged only over the last time step)
- ``mcm_rotate_acceptance`` - Fraction of rotation moves accepted (averaged only over the last time step)
- ``mcm_d`` - Maximum move displacement
- ``mcm_a`` - Maximum rotation move
- ``mcm_move_ratio`` - Probability of making a translation move (1- P(rotate move))
- ``mcm_overlap_count`` - Count of the number of particle-particle overlaps in the current system configuration

With non-interacting depletant (**implicit=True**), the following log quantities are available:

- ``mcm_fugacity`` - The current value of the depletant fugacity (in units of density, volume^-1)
- ``mcm_ntrial`` - The current number of configurational bias attempts per overlapping depletant
- ``mcm_insert_count`` - Number of depletants inserted per colloid
- ``mcm_reinsert_count`` - Number of overlapping depletants reinserted per colloid by configurational bias MC
- ``mcm_free_volume_fraction`` - Fraction of free volume to total sphere volume after a trial move has been proposed
  (sampled inside a sphere around the new particle position)
- ``mcm_overlap_fraction`` - Fraction of deplatants in excluded volume after trial move to depletants in free volume before move
- ``mcm_configurational_bias_ratio`` - Ratio of configurational bias attempts to depletant insertions

:py:class:`compute.free_volume` provides the following loggable quantities:
- ``mcm_free_volume`` - The free volume estimate in the simulation box obtained by MC sampling (in volume units)

:py:class:`update.boxmc` provides the following loggable quantities:

- ``mcm_boxmc_trial_count`` - Number of box changes attempted since the start of the boxmc updater
- ``mcm_boxmc_volume_acceptance`` - Fraction of volume/length change trials accepted (averaged from the start of the last run)
- ``mcm_boxmc_ln_volume_acceptance`` - Fraction of log(volume) change trials accepted (averaged from the start of the last run)
- ``mcm_boxmc_shear_acceptance`` - Fraction of shear trials accepted (averaged from the start of the last run)
- ``mcm_boxmc_aspect_acceptance`` - Fraction of aspect trials accepted (averaged from the start of the last run)
- ``mcm_boxmc_betaP`` Current value of the :math:`\beta p` value of the boxmc updater

:py:class:`update.muvt` provides the following loggable quantities.

- ``mcm_muvt_insert_acceptance`` - Fraction of particle insertions accepted (averaged from start of run)
- ``mcm_muvt_remove_acceptance`` - Fraction of particle removals accepted (averaged from start of run)
- ``mcm_muvt_volume_acceptance`` - Fraction of particle removals accepted (averaged from start of run)

:py:class:`update.clusters()` provides the following loggable quantities.

- ``mcm_clusters_moves`` - Fraction of cluster moves divided by the number of particles
- ``mcm_clusters_pivot_acceptance`` - Fraction of pivot moves accepted
- ``mcm_clusters_reflection_acceptance`` - Fraction of reflection moves accepted
- ``mcm_clusters_swap_acceptance`` - Fraction of swap moves accepted
- ``mcm_clusters_avg_size`` - Average cluster size

.. rubric:: Timestep definition

HOOMD-blue started as an MD code where **timestep** has a clear meaning. MC simulations are run
for timesteps. In exact terms, this means different things on the CPU and GPU and something slightly different when
using MPI. The behavior is approximately normalized so that user scripts do not need to drastically change
run() lengths when switching from one execution resource to another.

In the GPU implementation, one trial move is applied to a number of randomly chosen particles in each cell during one
timestep. The number of selected particles is ``nselect*ceil(avg particles per cell)`` where *nselect* is a user-chosen
parameter. The default value of *nselect* is 4, which achieves optimal performance for a wide variety of benchmarks.
Detailed balance is obeyed at the level of a timestep. In short: One timestep **is NOT equal** to one sweep,
but is approximately *nselect* sweeps, which is an overestimation.

In the single-threaded CPU implementation, one trial move is applied *nselect* times to each of the *N* particles
during one timestep. In parallel MPI runs, one trial moves is applied *nselect* times to each particle in the active
region. There is a small strip of inactive region near the boundaries between MPI ranks in the domain decomposition.
The trial moves are performed in a shuffled order so detailed balance is obeyed at the level of a timestep.
In short: One timestep **is approximately** *nselect* sweeps (*N* trial moves). In single-threaded runs, the
approximation is exact, but it is slightly underestimated in MPI parallel runs.

To approximate a fair comparison of dynamics between CPU and GPU timesteps, log the ``mcm_sweep``
quantity to get the number sweeps completed so far at each logged timestep.

See `J. A. Anderson et. al. 2016 <http://dx.doi.org/10.1016/j.cpc.2016.02.024>`_ for design and implementation details.

.. rubric:: Stability

:py:mod:`hoomd.mcm` is **stable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts that follow *documented* interfaces for functions and classes
will not require any modifications. **Maintainer:** Joshua A. Anderson
"""

# need to import all submodules defined in this directory
from hoomd.mcm import integrate
from hoomd.mcm import update
from hoomd.mcm import analyze
from hoomd.mcm import compute
from hoomd.mcm import util
from hoomd.mcm import field

# add HPMC article citation notice
import hoomd
_citation = hoomd.cite.article(cite_key='anderson2016',
                               author=['J A Anderson', 'M E Irrgang', 'S C Glotzer'],
                               title='Scalable Metropolis Monte Carlo for simulation of hard shapes',
                               journal='Computer Physics Communications',
                               volume=204,
                               pages='21--30',
                               month='July',
                               year='2016',
                               doi='10.1016/j.cpc.2016.02.024',
                               feature='HPMC')

if hoomd.context.bib is None:
    hoomd.cite._extra_default_entries.append(_citation)
else:
    hoomd.context.bib.add(_citation)
