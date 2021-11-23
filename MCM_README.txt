Compiling From Source:

cd ./Mechanical-Contraction-Method
mkdir ./build
cd ./build
pwd
export SOFTWARE_ROOT=path/to/Mechanical-Contraction-Method/build
cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python
ccmake . (set which modules to build)
make install

If you get an error about python not being able to import hoomd, make sure you're running using the same version of python you compiled against. If not, specify your compilation python in the cmake command with -DPYTHON_EXECUTABLE=/usr/bin/python3 or wherever your python is.

Running the simulation:

Initialize hoomd jobscript as usual, to invoke the MCM integrator using similar syntax to HPMC:

length=5
cap_radius=1

MCM=hoomd.mcm.integrate.convex_spheropolyhedron(d=0.1, a=0.1,seed=12345)
MCM.shape_param.set('A', vertices=[(0.0,0.0,-length/2.0),(0.0,0.0,length/2.0)],sweep_radius=cap_radius)

hoomd.run(100)
