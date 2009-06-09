#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the boost unit test module
#define BOOST_TEST_MODULE AngleForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "HarmonicAngleForceCompute.h"
#include "ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "HarmonicAngleForceComputeGPU.h"
#endif

#include "Initializers.h"

using namespace std;
using namespace boost;

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Global tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-1);
#else
const Scalar tol = 1e-2;
#endif

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<HarmonicAngleForceCompute>  (shared_ptr<SystemDefinition> sysdef)> angleforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void angle_force_basic_tests(angleforce_creator af_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
        
	/////////////////////////////////////////////////////////
	// start with the simplest possible test: 3 particles in a huge box with only one bond type !!!! NO ANGLES
	shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 1, 1, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
	
	ParticleDataArrays arrays = pdata_3->acquireReadWrite();
        arrays.x[0] = Scalar(-1.23); // put atom a at (-1,0,0.1)
        arrays.y[0] = Scalar(2.0);
        arrays.z[0] = Scalar(0.1);

	arrays.x[1] = arrays.y[1] = arrays.z[1] = Scalar(1.0); // put atom b at (0,0,0)

        arrays.x[2] = Scalar(1.0); // put atom c at (1,0,0.5)
        arrays.y[2] = 0.0;
        arrays.z[2] = Scalar(0.500);
  
        //printf(" Particle 1: x = %f  y = %f  z = %f \n", arrays.x[0], arrays.y[0], arrays.z[0]);
        //printf(" Particle 2: x = %f  y = %f  z = %f \n", arrays.x[1], arrays.y[1], arrays.z[1]);      
        //printf(" Particle 3: x = %f  y = %f  z = %f \n", arrays.x[2], arrays.y[2], arrays.z[2]);            
        //printf("\n");

	pdata_3->release();

	// create the angle force compute to check
	shared_ptr<HarmonicAngleForceCompute> fc_3 = af_creator(sysdef_3);
	fc_3->setParams(0, 1.0, 0.785398); // type=0, K=1.0,theta_0=pi/4=0.785398

	// compute the force and check the results
	fc_3->compute(0);
	ForceDataArrays force_arrays = fc_3->acquire();

	// check that the force is correct, it should be 0 since we haven't created any angles yet
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
	
	// add an angle and check again
	sysdef_3->getAngleData()->addAngle(Angle(0,0,1,2)); // add type 0 bewtween angle formed by atom 0-1-2
	fc_3->compute(1);

	
	// this time there should be a force
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.123368, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -0.626939, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -0.390920, tol);
        MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.158576, tol);
        MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);

	//MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	//MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0.564651,tol);	
	//MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.298813, tol);
	//MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 0.0000001, tol);

/*
        printf(" Particle 1: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[0], force_arrays.fy[0], force_arrays.fz[0]);
        printf(" Particle 2: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[1], force_arrays.fy[1], force_arrays.fz[1]);      
        printf(" Particle 3: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[2], force_arrays.fy[2], force_arrays.fz[2]);    
        printf(" Energy: 1 = %f  2 = %f  3 = %f \n\n", force_arrays.pe[0], force_arrays.pe[1], force_arrays.pe[2]);
        printf("\n");
*/

//        arrays = pdata_3->acquireReadWrite();


	
	// rearrange the two particles in memory and see if they are properly updated
	arrays = pdata_3->acquireReadWrite();


        arrays.x[1] = Scalar(-1.23); // put atom a at (-1,0,0.1)
        arrays.y[1] = Scalar(2.0);
        arrays.z[1] = Scalar(0.1);

	arrays.x[0] = arrays.y[0] = arrays.z[0] = Scalar(1.0); // put atom b at (0,0,0)

	arrays.tag[0] = 1;
	arrays.tag[1] = 0;
	arrays.rtag[0] = 1;
	arrays.rtag[1] = 0;
	pdata_3->release();

	// notify that we made the sort
	pdata_3->notifyParticleSort();
	// recompute at the same timestep, the forces should still be updated
	fc_3->compute(1);

	force_arrays = fc_3->acquire();
  
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.123368, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], -0.626939, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], -0.390920, tol);
        MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.158576, tol);
        MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
        //pdata_3->release();	

 



	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles across each boundary
	// also test more than one type of bond
        unsigned int num_angles_to_test = 3;
	shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 1, num_angles_to_test, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

	arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	pdata_6->release();
	
	shared_ptr<HarmonicAngleForceCompute> fc_6 = af_creator(sysdef_6);
	fc_6->setParams(0, 1.0, 0.785398);
	fc_6->setParams(1, 2.0, 1.46);
	//fc_6->setParams(2, 1.5, 1.68);
	
	sysdef_6->getAngleData()->addAngle(Angle(0, 0,1,2));
	sysdef_6->getAngleData()->addAngle(Angle(1, 3,4,5));
	//pdata_6->getAngleData()->addAngle(Angle(2, 3,4,5));
	
	fc_6->compute(0);
	// check that the forces are correctly computed
	force_arrays = fc_6->acquire();

        //printf(" \nParticle 1: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[0], force_arrays.fy[0], force_arrays.fz[0]);
        //printf(" Particle 2: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[1], force_arrays.fy[1], force_arrays.fz[1]);      
        //printf(" Particle 3: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[2], force_arrays.fy[2], force_arrays.fz[2]);    
        //printf(" Energy: 1 = %f  2 = %f  3 = %f \n", force_arrays.pe[0], force_arrays.pe[1], force_arrays.pe[2]);
        //printf(" Virial: 1 = %f  2 = %f  3 = %f \n", force_arrays.virial[0], force_arrays.virial[1], force_arrays.virial[2]);
        //printf(" \nParticle 4: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[3], force_arrays.fy[3], force_arrays.fz[3]);
        //printf(" Particle 5: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[4], force_arrays.fy[4], force_arrays.fz[4]);      
        //printf(" Particle 6: fx = %f  fy = %f  fz = %f \n", force_arrays.fx[5], force_arrays.fy[5], force_arrays.fz[5]);    
        //printf(" Energy: 4 = %f  5 = %f  6 = %f \n", force_arrays.pe[3], force_arrays.pe[4], force_arrays.pe[5]);
        //printf(" Virial: 4 = %f  5 = %f  6 = %f \n", force_arrays.virial[3], force_arrays.virial[4], force_arrays.virial[5]);

        //printf("\n");


	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -3.102127,tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.256618, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.102119, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 3.152144,tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.256618, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0.102119,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -0.050017, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.256618, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[3], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0.103030, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], -0.068223,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.400928, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[4], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], -5.586610,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], 0.068222, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 0.400928, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[4], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[5], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[5], 5.483580,tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[5], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[5], 0.400928, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[5], tol);



	// one more test: this one will test two things:
	// 1) That the forces are computed correctly even if the particles are rearranged in memory
	// and 2) That two forces can add to the same particle
	shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(100.0, 100.0, 100.0), 1, 1, 1, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();
	
	arrays = pdata_4->acquireReadWrite();
	// make a square of particles
	arrays.x[0] = 0.0; arrays.y[0] = 0.0; arrays.z[0] = 0.0;
	arrays.x[1] = 1.0; arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = 1.0; arrays.z[2] = 0.0;
	arrays.x[3] = 1.0; arrays.y[3] = 1.0; arrays.z[3] = 0.0;

	arrays.tag[0] = 2;
	arrays.tag[1] = 3;
	arrays.tag[2] = 0;
	arrays.tag[3] = 1;
	arrays.rtag[arrays.tag[0]] = 0;
	arrays.rtag[arrays.tag[1]] = 1;
	arrays.rtag[arrays.tag[2]] = 2;
	arrays.rtag[arrays.tag[3]] = 3;
	pdata_4->release();

	// build the bond force compute and try it out
	shared_ptr<HarmonicAngleForceCompute> fc_4 = af_creator(sysdef_4);
	fc_4->setParams(0, 1.5, 1.75);
	// only add bonds on the left, top, and bottom of the square
	sysdef_4->getAngleData()->addAngle(Angle(0, 0,1,2));
	sysdef_4->getAngleData()->addAngle(Angle(0, 1,2,3));
	sysdef_4->getAngleData()->addAngle(Angle(0, 0,1,3));

	fc_4->compute(0);
	force_arrays = fc_4->acquire();


	// the first particles shoul only have a force pulling them right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 2.893805, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.465228, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);

	// and the bottom left particle should have a force pulling up and to the right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 0.537611, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], -2.893805,tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.240643, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);

	// the bottom left particle should have a force pulling down 
	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 3.431415, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.240643,tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);

	// and the top left particle should have a force pulling up and to the left
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], -3.431415, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], -0.537611, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[3], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.473257, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);



	}





//! Compares the output of two HarmonicAngleForceComputes
void angle_force_comparison_tests(angleforce_creator af_creator1, angleforce_creator af_creator2, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	const unsigned int N = 1000;
	
	// create a particle system to sum forces on
	// just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
	RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
	
	shared_ptr<HarmonicAngleForceCompute> fc1 = af_creator1(sysdef);
	shared_ptr<HarmonicAngleForceCompute> fc2 = af_creator2(sysdef);
	fc1->setParams(0, Scalar(1.0), Scalar(1.348));
	fc2->setParams(0, Scalar(1.0), Scalar(1.348));

	// add angles
	for (unsigned int i = 0; i < N-2; i++)
		{
		sysdef->getAngleData()->addAngle(Angle(0, i, i+1,i+2));
		}
		
	// compute the forces
	fc1->compute(0);
	fc2->compute(0);
	
	// verify that the forces are identical (within roundoff errors)
	ForceDataArrays arrays1 = fc1->acquire();
	ForceDataArrays arrays2 = fc2->acquire();

        
	Scalar rough_tol = Scalar(3.0);

	for (unsigned int i = 0; i < N; i++)
		{
		BOOST_CHECK_CLOSE(arrays1.fx[i], arrays2.fx[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fy[i], arrays2.fy[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fz[i], arrays2.fz[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], rough_tol);
		BOOST_CHECK_SMALL(arrays1.virial[i], rough_tol);
		BOOST_CHECK_SMALL(arrays2.virial[i], rough_tol);
		}
        
	}


	


//! HarmonicAngleForceCompute creator for angle_force_basic_tests()
shared_ptr<HarmonicAngleForceCompute> base_class_af_creator(shared_ptr<SystemDefinition> sysdef)
	{
	return shared_ptr<HarmonicAngleForceCompute>(new HarmonicAngleForceCompute(sysdef));
	}

#ifdef ENABLE_CUDA
//! AngleForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicAngleForceCompute> gpu_af_creator(shared_ptr<SystemDefinition> sysdef)
	{
	return shared_ptr<HarmonicAngleForceCompute>(new HarmonicAngleForceComputeGPU(sysdef));
	}
#endif

//! boost test case for angle forces on the CPU
BOOST_AUTO_TEST_CASE( HarmonicAngleForceCompute_basic )
	{
        printf(" IN BOOST_AUTO_TEST_CASE: CPU \n");
	angleforce_creator af_creator = bind(base_class_af_creator, _1);
	angle_force_basic_tests(af_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}

#ifdef ENABLE_CUDA
//! boost test case for angle forces on the GPU
BOOST_AUTO_TEST_CASE( HarmonicAngleForceComputeGPU_basic )
	{
        printf(" IN BOOST_AUTO_TEST_CASE: GPU \n");
	angleforce_creator af_creator = bind(gpu_af_creator, _1);
	angle_force_basic_tests(af_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	}

	
//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( HarmonicAngleForceComputeGPU_compare )
	{
	angleforce_creator af_creator_gpu = bind(gpu_af_creator, _1);
	angleforce_creator af_creator = bind(base_class_af_creator, _1);
	angle_force_comparison_tests(af_creator, af_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	}
	
//! boost test case for comparing calculation on the CPU to multi-gpu ones
BOOST_AUTO_TEST_CASE( HarmonicAngleForce_MultiGPU_compare)
	{
	vector<unsigned int> gpu_list;
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	ExecutionConfiguration exec_conf(ExecutionConfiguration::GPU, gpu_list);
	
	angleforce_creator af_creator_gpu = bind(gpu_af_creator, _1);
	angleforce_creator af_creator = bind(base_class_af_creator, _1);
	angle_force_comparison_tests(af_creator, af_creator_gpu, exec_conf);
	}

#endif
