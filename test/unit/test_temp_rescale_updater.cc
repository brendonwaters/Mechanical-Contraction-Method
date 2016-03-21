/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// this include is necessary to get MPI included before anything else to support intel MPI
#include "ExecutionConfiguration.h"

#include <iostream>

#include <boost/shared_ptr.hpp>

#include "ComputeThermo.h"
#include "TempRescaleUpdater.h"

#ifdef ENABLE_CUDA
#include "ComputeThermoGPU.h"
#endif

#include <math.h>

using namespace std;
using namespace boost;

//! label the boost test module
#define BOOST_TEST_MODULE TempRescaleUpdaterTests
#include "boost_utf_configure.h"


/*! \file temp_rescale_updater_test.cc
    \brief Unit tests for the ComputeThermo and TempRescaleUpdater classes.
    \ingroup unit_tests
*/

//! boost test case to verify proper operation of ComputeThermo
BOOST_AUTO_TEST_CASE( ComputeThermo_basic )
    {
    // verify that we can constructe a TempCompute properly
    // create a simple particle data to test with
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_vel.data[0].x = 1.0; h_vel.data[0].y = 2.0; h_vel.data[0].z = 3.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    h_vel.data[1].x = 4.0; h_vel.data[1].y = 5.0; h_vel.data[1].z = 6.0;
    }

    // construct a TempCompute and see that everything is set properly
    boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    boost::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    boost::shared_ptr<ComputeThermo> tc(new ComputeThermo(sysdef, group_all));

    // check that we can actually compute temperature
    tc->setNDOF(3*pdata->getN());
    tc->compute(0);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 15.1666666666666666666667, tol);
    }

#ifdef ENABLE_CUDA
//! boost test case to verify proper operation of ComputeThermoGPU
BOOST_AUTO_TEST_CASE( ComputeThermoGPU_basic )
    {
    // verify that we can constructe a TempCompute properly
    // create a simple particle data to test with
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_vel.data[0].x = 1.0; h_vel.data[0].y = 2.0; h_vel.data[0].z = 3.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    h_vel.data[1].x = 4.0; h_vel.data[1].y = 5.0; h_vel.data[1].z = 6.0;
    }

    // construct a TempCompute and see that everything is set properly
    boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    boost::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    boost::shared_ptr<ComputeThermoGPU> tc(new ComputeThermoGPU(sysdef, group_all));

    // check that we can actually compute temperature
    tc->setNDOF(3*pdata->getN());
    tc->compute(0);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 15.1666666666666666666667, tol);
    }
#endif

//! boost test case to verify proper operation of TempRescaleUpdater
BOOST_AUTO_TEST_CASE( TempRescaleUpdater_basic )
    {
    // create a simple particle data to test with
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_vel.data[0].x = 1.0; h_vel.data[0].y = 2.0; h_vel.data[0].z = 3.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    h_vel.data[1].x = 4.0; h_vel.data[1].y = 5.0; h_vel.data[1].z = 6.0;
    }

    // construct a Computethermo for the updater
    boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    boost::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    boost::shared_ptr<ComputeThermo> tc(new ComputeThermo(sysdef, group_all));


    // variant T for the rescaler
    boost::shared_ptr<VariantConst> T_variant(new VariantConst(1.2));

    // construct the updater and make sure everything is set properly
    boost::shared_ptr<TempRescaleUpdater> rescaler(new TempRescaleUpdater(sysdef, tc, T_variant));

    // run the updater and check the new temperature
    rescaler->update(0);
    tc->compute(1);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 1.2, tol);

    // check that the setT method works
    boost::shared_ptr<VariantConst> T_variant2(new VariantConst(2.0));
    rescaler->setT(T_variant2);
    rescaler->update(1);
    tc->compute(2);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 2.0, tol);
    }

