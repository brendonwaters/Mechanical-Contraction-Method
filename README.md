<<<<<<< HEAD
=======
# Mechanical-Contraction-Method
Particle simulation package based on HOOMD-Blue

The Mechanical-Contraction-Method (MCM) is a method for creating dense random packings of spherocylinder particles. By taking an intial random configuration of particles at low densities and compressing them while iteratively removing overlaps between neighboring particles the MCM algorithm rapidly increases the density of the system while retaining its initial entropy. 

Compiling Hoomd with MCM From Source:

$ cd Mechanical-Contraction-Method
$ mkdir build
$ cd build

Set the SOFTWARE_ROOT environment variable:
$ export SOFTWARE_ROOT=path/to/Mechanical-Contraction-Method/build

Initialize the cmake scripts:
$ cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python

Use ccmake GUI to set which modules you want to compile (optional) :

$ ccmake . 

And finally:
$ make install

If you get an error about python not being able to import hoomd, make sure you're running using the same version of python you compiled against. If not, specify your compilation python in the cmake command with -DPYTHON_EXECUTABLE=/usr/bin/python3 or wherever your python is.

Running the simulation:

Initialize hoomd jobscript as usual, to invoke the MCM integrator using similar syntax to HPMC:

length=5
cap_radius=1

MCM=hoomd.mcm.integrate.convex_spheropolyhedron(d=0.1, a=0.1,seed=12345)
MCM.shape_param.set('A', vertices=[(0.0,0.0,-length/2.0),(0.0,0.0,length/2.0)],sweep_radius=cap_radius)

hoomd.run(100)

Citation for the MCM algorithm: 

S. R. Williams and A. P. Philipse, Random packings of
spheres and spherocylinders simulated by 
mechanical contraction, 
Phys. Rev. E 67, 051301 (2003).

This implementation of the MCM created by Brendon Waters in part for this work:

Shiva Pokhrel, Brendon Waters, Solveig Felton, Zhi-Feng Huang, and Boris Nadgorny. 
Percolation in metal-insulator composites of randomly packed 
spherocylindrical nanoparticles. Phys. Rev. B, 103:134110, Apr 2021. 
doi: 10.1103/PhysRevB.103.134110. 
URL https://link.aps.org/doi/10.1103/PhysRevB.103.
134110. 
https://arxiv.org/abs/2011.08124



# HOOMD-blue

HOOMD-blue is a general purpose particle simulation toolkit. It performs hard particle Monte Carlo simulations
of a variety of shape classes, and molecular dynamics simulations of particles with a range of pair, bond, angle,
and other potentials. HOOMD-blue runs fast on NVIDIA GPUs, and can scale across
many nodes. For more information, see the [HOOMD-blue website](http://glotzerlab.engin.umich.edu/hoomd-blue).

# Tutorial

[Read the HOOMD-blue tutorial online](http://nbviewer.jupyter.org/github/joaander/hoomd-examples/blob/master/index.ipynb).

## Installing HOOMD-blue

Official binaries of HOOMD-blue are available via [conda](http://conda.pydata.org/docs/) through
the [glotzer channel](https://anaconda.org/glotzer).
To install HOOMD-blue, first download and install
[miniconda](http://conda.pydata.org/miniconda.html) following [conda's instructions](http://conda.pydata.org/docs/install/quick.html).
Then add the `glotzer` channel and install HOOMD-blue:

```bash
$ conda config --add channels glotzer
$ conda install hoomd cudatoolkit=8.0
```

Conda does not properly pin the CUDA toolkit version in the dependencies, so you must explicitly request
`cudatoolkit=8.0`.

## Compiling HOOMD-blue

Use cmake to configure an out of source build and make to build hoomd.

```bash
mkdir build
cd build
cmake ../
make -j20
```

To run out of the build directory, add the build directory to your `PYTHONPATH`:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
```

For more detailed instructions, [see the documentation](http://hoomd-blue.readthedocs.io/en/stable/compiling.html).

### Prerequisites

 * Required:
     * Python >= 2.7
     * numpy >= 1.7
     * CMake >= 2.8.0
     * C++ 11 capable compiler (tested with gcc 4.8, 4.9, 5.4, 6.4, clang 3.8, 5.0)
 * Optional:
     * NVIDIA CUDA Toolkit >= 7.0
     * Intel Threaded Building Blocks >= 4.3
     * MPI (tested with OpenMPI, MVAPICH)
     * sqlite3

## Job scripts

HOOMD-blue job scripts are python scripts. You can control system initialization, run protocol, analyze simulation data,
or develop complex workflows all with python code in your job.

Here is a simple example.

```python
import hoomd
from hoomd import md
hoomd.context.initialize()

# create a 10x10x10 square lattice of particles with name A
hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0, type_name='A'), n=10)
# specify Lennard-Jones interactions between particle pairs
nl = md.nlist.cell()
lj = md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# integrate at constant temperature
all = hoomd.group.all();
md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=all, kT=1.2, seed=4)
# run 10,000 time steps
hoomd.run(10e3)
```

Save this as `lj.py` and run with `python lj.py`.

## Reference Documentation

Read the [reference documentation on readthedocs](http://hoomd-blue.readthedocs.io).

## Change log

See [ChangeLog.md](ChangeLog.md).

## Contributing to HOOMD-blue.

See [CONTRIBUTING.md](CONTRIBUTING.md).
>>>>>>> master
