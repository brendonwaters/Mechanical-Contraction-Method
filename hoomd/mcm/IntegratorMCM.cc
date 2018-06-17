// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorMCM.h"

namespace py = pybind11;

#include "hoomd/VectorMath.h"
#include <sstream>

using namespace std;

/*! \file IntegratorMCM.cc
    \brief Definition of common methods for MCM integrators
*/

namespace mcm
{

IntegratorMCM::IntegratorMCM(std::shared_ptr<SystemDefinition> sysdef,
                               unsigned int seed)
    : Integrator(sysdef, 0.005), m_seed(seed),  m_move_ratio(32768), m_nselect(4),
      m_nominal_width(1.0), m_extra_ghost_width(0), m_external_base(NULL), m_patch_log(false),
      m_past_first_run(false)
      #ifdef ENABLE_MPI
      ,m_communicator_ghost_width_connected(false),
      m_communicator_flags_connected(false)
      #endif
    {
    m_exec_conf->msg->notice(5) << "Constructing IntegratorMCM" << endl;

    // broadcast the seed from rank 0 to all other ranks.
    #ifdef ENABLE_MPI
        if(this->m_pdata->getDomainDecomposition())
            bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif

    GPUArray<mcm_counters_t> counters(1, this->m_exec_conf);
    m_count_total.swap(counters);

    GPUVector<Scalar> d(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_d.swap(d);

    GPUVector<Scalar> a(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_a.swap(a);

    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::overwrite);
    //set default values
    for(unsigned int typ=0; typ < this->m_pdata->getNTypes(); typ++)
      {
      h_d.data[typ]=0.1;
      h_a.data[typ]=0.1;
      }

    // Connect to number of types change signal
    m_pdata->getNumTypesChangeSignal().connect<IntegratorMCM, &IntegratorMCM::slotNumTypesChange>(this);

    resetStats();
    }

IntegratorMCM::~IntegratorMCM()
    {
    m_exec_conf->msg->notice(5) << "Destroying IntegratorMCM" << endl;
    m_pdata->getNumTypesChangeSignal().disconnect<IntegratorMCM, &IntegratorMCM::slotNumTypesChange>(this);

    #ifdef ENABLE_MPI
    if (m_communicator_ghost_width_connected)
        m_comm->getGhostLayerWidthRequestSignal().disconnect<IntegratorMCM, &IntegratorMCM::getGhostLayerWidth>(this);
    if (m_communicator_flags_connected)
        m_comm->getCommFlagsRequestSignal().disconnect<IntegratorMCM, &IntegratorMCM::getCommFlags>(this);
    #endif
    }


void IntegratorMCM::slotNumTypesChange()
    {
    // old size of arrays
    unsigned int old_ntypes = m_a.size();
    assert(m_a.size() == m_d.size());

    unsigned int ntypes = m_pdata->getNTypes();

    m_a.resize(ntypes);
    m_d.resize(ntypes);

    //set default values for newly added types
    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::readwrite);
    for(unsigned int typ=old_ntypes; typ < ntypes; typ++)
        {
        h_d.data[typ]=0.1;
        h_a.data[typ]=0.1;
        }
    }

/*! IntegratorMCM provides:
    - mcm_sweep (Number of sweeps completed)
    - mcm_translate_acceptance (Ratio of translation moves accepted in the last step)
    - mcm_rotate_acceptance (Ratio of rotation moves accepted in the last step)
    - mcm_d (maximum move displacement)
    - mcm_a (maximum rotation move)
    - mcm_d_<typename> (maximum move displacement by type)
    - mcm_a_<typename> (maximum rotation move by type)
    - mcm_move_ratio (ratio of translation moves to rotate moves)
    - mcm_overlap_count (count of the number of particle-particle overlaps)

    \returns a list of provided quantities
*/
std::vector< std::string > IntegratorMCM::getProvidedLogQuantities()
    {
    // start with the integrator provided quantities
    std::vector< std::string > result = Integrator::getProvidedLogQuantities();
    // then add ours
    result.push_back("mcm_sweep");
    result.push_back("mcm_translate_acceptance");
    result.push_back("mcm_rotate_acceptance");
    result.push_back("mcm_d");
    result.push_back("mcm_a");
    result.push_back("mcm_move_ratio");
    result.push_back("mcm_overlap_count");
    for (unsigned int typ=0; typ<m_pdata->getNTypes();typ++)
      {
      ostringstream tmp_str0;
      tmp_str0<<"mcm_d_"<<m_pdata->getNameByType(typ);
      result.push_back(tmp_str0.str());

      ostringstream tmp_str1;
      tmp_str1<<"mcm_a_"<<m_pdata->getNameByType(typ);
      result.push_back(tmp_str1.str());
      }
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \return the requested log quantity.
*/
Scalar IntegratorMCM::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    mcm_counters_t counters = getCounters(2);

    if (quantity == "mcm_sweep")
        {
        mcm_counters_t counters_total = getCounters(0);
        return double(counters_total.getNMoves()) / double(m_pdata->getNGlobal());
        }
    else if (quantity == "mcm_translate_acceptance")
        {
        return counters.getTranslateAcceptance();
        }
    else if (quantity == "mcm_rotate_acceptance")
        {
        return counters.getRotateAcceptance();
        }
    else if (quantity == "mcm_d")
        {
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
        return h_d.data[0];
        }
    else if (quantity == "mcm_a")
        {
        ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);
        return h_a.data[0];
        }
    else if (quantity == "mcm_move_ratio")
        {
        return getMoveRatio();
        }
    else if (quantity == "mcm_overlap_count")
        {
        return countOverlaps(timestep, false);
        }
    else
        {
        //loop over per particle move size quantities
        for (unsigned int typ=0; typ<m_pdata->getNTypes();typ++)
          {
          ostringstream tmp_str0;
          tmp_str0<<"mcm_d_"<<m_pdata->getNameByType(typ);
          if (quantity==tmp_str0.str())
            {
            ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
            return h_d.data[typ];
            }

          ostringstream tmp_str1;
          tmp_str1<<"mcm_a_"<<m_pdata->getNameByType(typ);
          if (quantity==tmp_str1.str())
            {
            ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);
            return h_a.data[typ];
            }
          }

        //nothing found -> pass on to integrator
        return Integrator::getLogValue(quantity, timestep);
        }
    }

/*! \returns True if the particle orientations are normalized
*/
bool IntegratorMCM::checkParticleOrientations()
    {
    // get the orientations data array
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    bool result = true;

    // loop through particles and return false if any is out of norm
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        quat<Scalar> o(h_orientation.data[i]);
        if (fabs(Scalar(1.0) - norm2(o)) > 1e-3)
            {
            m_exec_conf->msg->notice(2) << "Particle " << h_tag.data[i] << " has an unnormalized orientation" << endl;
            result = false;
            }
        }

    #ifdef ENABLE_MPI
    unsigned int result_int = (unsigned int)result;
    unsigned int result_reduced;
    MPI_Reduce(&result_int, &result_reduced, 1, MPI_UNSIGNED, MPI_LOR, 0, m_exec_conf->getMPICommunicator());
    result = bool(result_reduced);
    #endif

    return result;
    }

/*! Set new box with particle positions scaled from previous box
    and check for overlaps

    \param newBox new box dimensions

    \note The particle positions and the box dimensions are updated in any case, even if the
    new box dimensions result in overlaps. To restore old particle positions,
    they have to be backed up before calling this method.

    \returns false if resize results in overlaps
*/
bool IntegratorMCM::attemptBoxResize(unsigned int timestep, const BoxDim& new_box)
    {
    unsigned int N = m_pdata->getN();

    // Get old and new boxes;
    BoxDim curBox = m_pdata->getGlobalBox();

    // Use lexical scope block to make sure ArrayHandles get cleaned up
        {
        // Get particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

        // move the particles to be inside the new box
        for (unsigned int i = 0; i < N; i++)
            {
            Scalar3 old_pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

            // obtain scaled coordinates in the old global box
            Scalar3 f = curBox.makeFraction(old_pos);

            // scale particles
            Scalar3 scaled_pos = new_box.makeCoordinates(f);
            h_pos.data[i].x = scaled_pos.x;
            h_pos.data[i].y = scaled_pos.y;
            h_pos.data[i].z = scaled_pos.z;
            }
        } // end lexical scope

    m_pdata->setGlobalBox(new_box);

    // we have moved particles, communicate those changes
    this->communicate(false);

    // check overlaps
    return !this->countOverlaps(timestep, true);
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    IntegratorMCM maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
mcm_counters_t IntegratorMCM::getCounters(unsigned int mode)
    {
    ArrayHandle<mcm_counters_t> h_counters(m_count_total, access_location::host, access_mode::read);
    mcm_counters_t result;

    if (mode == 0)
        result = h_counters.data[0];
    else if (mode == 1)
        result = h_counters.data[0] - m_count_run_start;
    else
        result = h_counters.data[0] - m_count_step_start;

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // MPI Reduction to total result values on all nodes.
        MPI_Allreduce(MPI_IN_PLACE, &result.translate_accept_count, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.translate_reject_count, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.rotate_accept_count, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.rotate_reject_count, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.overlap_checks, 1, MPI_LONG_LONG_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.overlap_err_count, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

void export_IntegratorMCM(py::module& m)
    {
   py::class_<IntegratorMCM, std::shared_ptr< IntegratorMCM > >(m, "IntegratorMCM", py::base<Integrator>())
    .def(py::init< std::shared_ptr<SystemDefinition>, unsigned int >())
    .def("setD", &IntegratorMCM::setD)
    .def("setA", &IntegratorMCM::setA)
    .def("setMoveRatio", &IntegratorMCM::setMoveRatio)
    .def("setNSelect", &IntegratorMCM::setNSelect)
    .def("getD", &IntegratorMCM::getD)
    .def("getA", &IntegratorMCM::getA)
    .def("getMoveRatio", &IntegratorMCM::getMoveRatio)
    .def("getNSelect", &IntegratorMCM::getNSelect)
    .def("getMaxCoreDiameter", &IntegratorMCM::getMaxCoreDiameter)
    .def("countOverlaps", &IntegratorMCM::countOverlaps)
    .def("checkParticleOrientations", &IntegratorMCM::checkParticleOrientations)
    .def("getMPS", &IntegratorMCM::getMPS)
    .def("getCounters", &IntegratorMCM::getCounters)
    .def("communicate", &IntegratorMCM::communicate)
    .def("slotNumTypesChange", &IntegratorMCM::slotNumTypesChange)
    .def("setDeterministic", &IntegratorMCM::setDeterministic)
    .def("disablePatchEnergyLogOnly", &IntegratorMCM::disablePatchEnergyLogOnly)
    ;

   py::class_< mcm_counters_t >(m, "mcm_counters_t")
    .def_readwrite("translate_accept_count", &mcm_counters_t::translate_accept_count)
    .def_readwrite("translate_reject_count", &mcm_counters_t::translate_reject_count)
    .def_readwrite("rotate_accept_count", &mcm_counters_t::rotate_accept_count)
    .def_readwrite("rotate_reject_count", &mcm_counters_t::rotate_reject_count)
    .def_readwrite("overlap_checks", &mcm_counters_t::overlap_checks)
    .def("getTranslateAcceptance", &mcm_counters_t::getTranslateAcceptance)
    .def("getRotateAcceptance", &mcm_counters_t::getRotateAcceptance)
    .def("getNMoves", &mcm_counters_t::getNMoves)
    ;
    }

} // end namespace mcm
