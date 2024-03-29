// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _INTEGRATOR_MCM_MONO_H_
#define _INTEGRATOR_MCM_MONO_H_

/*! \file IntegratorMCMMono.h
    \brief Declaration of IntegratorMCM
*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <random>

#include "hoomd/Integrator.h"
#include "MCMPrecisionSetup.h"
#include "IntegratorMCM.h"
#include "Moves.h"
#include "AABBTree.h"
#include "GSDMCMSchema.h"
#include "hoomd/Index1D.h"
#include "hoomd/extern/Eigen/Eigen/Core"
#include "hoomd/extern/Eigen/Eigen/Dense"

#include "hoomd/managed_allocator.h"

#include "hoomd/SnapshotSystemData.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace mcm
{

namespace detail
{

//! Helper class to manage shuffled update orders
/*! Stores an update order from 0 to N-1, inclusive, and can be resized. shuffle() shuffles the order of elements
    to a new random permutation. operator [i] gets the index of the item at order i in the current shuffled sequence.

    \ingroup mcm_data_structs
*/
class UpdateOrder
    {
    public:
        //! Constructor
        /*! \param seed Random number seed
            \param N number of integers to shuffle
        */
        UpdateOrder(unsigned int seed, unsigned int N=0)
            : m_seed(seed)
            {
            resize(N);
            }

        //! Resize the order
        /*! \param N new size
            \post The order is 0, 1, 2, ... N-1
        */
    void resize(unsigned int N)
            {
            // initialize the update order
            m_update_order.resize(N);
            for (unsigned int i = 0; i < N; i++)
                m_update_order[i] = i;
            }

        //! Shuffle the order
        /*! \param timestep Current timestep of the simulation
            \note \a timestep is used to seed the RNG, thus assuming that the order is shuffled only once per
            timestep.
        */
        void shuffle(unsigned int timestep, unsigned int select = 0)
            {
            hoomd::detail::Saru rng(timestep, m_seed+select, 0xfa870af6);
            float r = rng.f();

            // reverse the order with 1/2 probability
            if (r > 0.5f)
                {
                unsigned int N = m_update_order.size();
                for (unsigned int i = 0; i < N; i++)
                    m_update_order[i] = N - i - 1;
                }
            else
                {
                unsigned int N = m_update_order.size();
                for (unsigned int i = 0; i < N; i++)
                    m_update_order[i] = i;
                }
            }

        //! Access element of the shuffled order
        unsigned int operator[](unsigned int i)
            {
            return m_update_order[i];
            }
    private:
        unsigned int m_seed;                       //!< Random number seed
        std::vector<unsigned int> m_update_order; //!< Update order
    };

}; // end namespace detail

//! MCM on systems of mono-disperse shapes
/*! Implement hard particle monte carlo for a single type of shape on the CPU.

    TODO: I need better documentation

    \ingroup mcm_integrators
*/
template < class Shape >
class IntegratorMCMMono : public IntegratorMCM
    {
    public:
        //! Param type from the shape
        //! Each shape has a param_type member that contain
        //! shape-specific descriptors(radius, vertices, etc)
        typedef typename Shape::param_type param_type;

        //! Constructor
        IntegratorMCMMono(std::shared_ptr<SystemDefinition> sysdef,
                      unsigned int seed);

        virtual ~IntegratorMCMMono()
            {
            if (m_aabbs != NULL)
                free(m_aabbs);
            m_pdata->getBoxChangeSignal().template disconnect<IntegratorMCMMono<Shape>, &IntegratorMCMMono<Shape>::slotBoxChanged>(this);
            m_pdata->getParticleSortSignal().template disconnect<IntegratorMCMMono<Shape>, &IntegratorMCMMono<Shape>::slotSorted>(this);
            }

        virtual void printStats();

        virtual bool attemptBoxResize(unsigned int timestep, const BoxDim& new_box);

        virtual void resetStats();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Returns ID and distance to near neighbors of particle i
        virtual std::vector<std::vector<Scalar>> getContactDistance(unsigned int i);


        //! Get the maximum particle diameter
        virtual Scalar getMaxCoreDiameter();

        //! Get the minimum particle diameter
        virtual OverlapReal getMinCoreDiameter();

        //! Set the pair parameters for a single type
        virtual void setParam(unsigned int typ, const param_type& param);

        //! Set elements of the interaction matrix
        virtual void setOverlapChecks(unsigned int typi, unsigned int typj, bool check_overlaps);

        //! Set the external field for the integrator
        void setExternalField(std::shared_ptr< ExternalFieldMono<Shape> > external)
            {
            m_external = external;
            this->m_external_base = (ExternalField*)external.get();
            }

        //! Get a list of logged quantities
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Get the particle parameters
        virtual std::vector<param_type, managed_allocator<param_type> >& getParams()
            {
            return m_params;
            }

        //! Get the interaction matrix
        virtual const GPUArray<unsigned int>& getInteractionMatrix()
            {
            return m_overlaps;
            }

        //! Get the indexer for the interaction matrix
        virtual const Index2D& getOverlapIndexer()
            {
            return m_overlap_idx;
            }

        //! Count overlaps with the option to exit early at the first detected overlap
        virtual unsigned int countOverlaps(unsigned int timestep, bool early_exit);

        //! Return a vector that is an unwrapped overlap map
        virtual std::vector<bool> mapOverlaps();

        //! Return a python list that is an unwrapped overlap map
        virtual pybind11::list PyMapOverlaps();

        //! Return the requested ghost layer width
        virtual Scalar getGhostLayerWidth(unsigned int)
            {
            Scalar ghost_width = m_nominal_width + m_extra_ghost_width;
            m_exec_conf->msg->notice(9) << "IntegratorMCMMono: ghost layer width of " << ghost_width << std::endl;
            return ghost_width;
            }

        #ifdef ENABLE_MPI
        //! Return the requested communication flags for ghost particles
        virtual CommFlags getCommFlags(unsigned int)
            {
            CommFlags flags(0);
            flags[comm_flag::position] = 1;
            flags[comm_flag::tag] = 1;

            std::ostringstream o;
            o << "IntegratorMCMMono: Requesting communication flags for pos tag ";
            if (m_hasOrientation)
                {
                flags[comm_flag::orientation] = 1;
                o << "orientation ";
                }

            if (m_patch)
                {
                flags[comm_flag::diameter] = 1;
                flags[comm_flag::charge] = 1;
                o << "diameter charge";
                }

            m_exec_conf->msg->notice(9) << o.str() << std::endl;
            return flags;
            }
        #endif

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep)
            {
            // base class method
            IntegratorMCM::prepRun(timestep);

                {
                // for p in params, if Shape dummy(q_dummy, params).hasOrientation() then m_hasOrientation=true
                m_hasOrientation = false;
                quat<Scalar> q(make_scalar4(1,0,0,0));
                for (unsigned int i=0; i < m_pdata->getNTypes(); i++)
                    {
                    Shape dummy(q, m_params[i]);
                    if (dummy.hasOrientation())
                        m_hasOrientation = true;
                    }
                }
            updateCellWidth(); // make sure the cell width is up-to-date and forces a rebuild of the AABB tree and image list

            communicate(true);
            }

        //! Communicate particles
        virtual void communicate(bool migrate)
            {
            // migrate and exchange particles
            #ifdef ENABLE_MPI
            if (m_comm)
                {
                // this is kludgy but necessary since we are calling the communications methods directly
                m_comm->setFlags(getCommFlags(0));

                if (migrate)
                    m_comm->migrateParticles();
                else
                    m_pdata->removeAllGhostParticles();

                m_comm->exchangeGhosts();

                m_aabb_tree_invalid = true;
                }
            #endif
            }

        //! Return true if anisotropic particles are present
        virtual bool hasOrientation() { return m_hasOrientation; }

        //! Compute the energy due to patch interactions
        /*! \param timestep the current time n
         * \returns the total patch energy
         */
        virtual float computePatchEnergy(unsigned int timestep);

        //! Build the AABB tree (if needed)
        const detail::AABBTree& buildAABBTree();

        //! Make list of image indices for boxes to check in small-box mode
        const std::vector<vec3<Scalar> >& updateImageList();

        //! Return list of integer shift vectors for periodic images
        const std::vector<int3>& getImageHKL()
            {
            updateImageList();
            return m_image_hkl;
            }

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange();

        void invalidateAABBTree(){ m_aabb_tree_invalid = true; }

        //! Method that is called whenever the GSD file is written if connected to a GSD file.
        int slotWriteGSD(gsd_handle&, std::string name) const;

        //! Method that is called to connect to the gsd write state signal
        void connectGSDSignal(std::shared_ptr<GSDDumpWriter> writer, std::string name);

        //! Method that is called to connect to the gsd write state signal
        bool restoreStateGSD(std::shared_ptr<GSDReader> reader, std::string name);

        void increment_mc_attempts()
            {
            n_mc+=1;
            }

    protected:
        std::vector<param_type, managed_allocator<param_type> > m_params;   //!< Parameters for each particle type on GPU
        GPUArray<unsigned int> m_overlaps;          //!< Interaction matrix (0/1) for overlap checks
        detail::UpdateOrder m_update_order;         //!< Update order
        bool m_image_list_is_initialized;                    //!< true if image list has been used
        bool m_image_list_valid;                             //!< image list is invalid if the box dimensions or particle parameters have changed.
        std::vector<vec3<Scalar> > m_image_list;             //!< List of potentially interacting simulation box images
        std::vector<int3> m_image_hkl;               //!< List of potentially interacting simulation box images (integer shifts)
        unsigned int m_image_list_rebuilds;                  //!< Number of times the image list has been rebuilt
        bool m_image_list_warning_issued;                    //!< True if the image list warning has been issued
        bool m_hkl_max_warning_issued;                       //!< True if the image list size warning has been issued
        bool m_hasOrientation;                               //!< true if there are any orientable particles in the system

        std::shared_ptr< ExternalFieldMono<Shape> > m_external;//!< External Field
        detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for overlap checks
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
        unsigned int m_aabbs_capacity;              //!< Capacity of m_aabbs list
        bool m_aabb_tree_invalid;                   //!< Flag if the aabb tree has been invalidated

        Scalar m_extra_image_width;                 //! Extra width to extend the image list

        Index2D m_overlap_idx;                      //!!< Indexer for interaction matrix

        bool max_density = false;                   // set true if MCM has reached density maximum, attempts_cutoff exceeded
        bool needs_mc = false;                      // triggers monte carlo moves after mcm compression cycle
        const unsigned int mc_cutoff=10;
        unsigned int n_mc=0;
        bool done=false;
        double a_max=0;
        double small=1e-3;
        double avg_contacts=0;

        //! Set the nominal width appropriate for looped moves
        virtual void updateCellWidth();

        //! Grow the m_aabbs list
        virtual void growAABBList(unsigned int N);

        //! Limit the maximum move distances
        virtual void limitMoveDistances();

        //! callback so that the box change signal can invalidate the image list
        virtual void slotBoxChanged()
            {
            m_image_list_valid = false;
            // changing the box does not necessarily invalidate the AABB tree - however, practically
            // anything that changes the box (i.e. NPT, box_resize) is also moving the particles,
            // so use it as a sign to rebuild the AABB tree
            m_aabb_tree_invalid = true;
            }

        //! callback so that the particle sort signal can invalidate the AABB tree
        virtual void slotSorted()
            {
            m_aabb_tree_invalid = true;
            }

        //! Write pairing data to file
        void writePairs(Scalar contactFactor);

        //! Use diffusion coefficient to calculate system bulk conductivity
        void diffuseConductivity(Scalar contactFactor);
    };

template <class Shape>
IntegratorMCMMono<Shape>::IntegratorMCMMono(std::shared_ptr<SystemDefinition> sysdef,
                                                   unsigned int seed)
            : IntegratorMCM(sysdef, seed),
              m_update_order(seed+m_exec_conf->getRank(), m_pdata->getN()),
              m_image_list_is_initialized(false),
              m_image_list_valid(false),
              m_hasOrientation(true),
              m_extra_image_width(0.0)
    {
    // allocate the parameter storage
    m_params = std::vector<param_type, managed_allocator<param_type> >(m_pdata->getNTypes(), param_type(), managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

    m_overlap_idx = Index2D(m_pdata->getNTypes());
    GPUArray<unsigned int> overlaps(m_overlap_idx.getNumElements(), m_exec_conf);
    m_overlaps.swap(overlaps);

    // Connect to the BoxChange signal
    m_pdata->getBoxChangeSignal().template connect<IntegratorMCMMono<Shape>, &IntegratorMCMMono<Shape>::slotBoxChanged>(this);
    m_pdata->getParticleSortSignal().template connect<IntegratorMCMMono<Shape>, &IntegratorMCMMono<Shape>::slotSorted>(this);

    m_image_list_rebuilds = 0;
    m_image_list_warning_issued = false;
    m_hkl_max_warning_issued = false;

    m_aabbs = NULL;
    m_aabbs_capacity = 0;
    m_aabb_tree_invalid = true;
    }


template<class Shape>
std::vector< std::string > IntegratorMCMMono<Shape>::getProvidedLogQuantities()
    {
    // start with the integrator provided quantities
    std::vector< std::string > result = IntegratorMCM::getProvidedLogQuantities();
    // then add ours
    if(m_patch)
        {
        result.push_back("mcm_patch_energy");
        result.push_back("mcm_patch_rcut");
        }

    return result;
    }

template<class Shape>
Scalar IntegratorMCMMono<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "mcm_patch_energy")
        {
        if (m_patch)
            {
            return computePatchEnergy(timestep);
            }
        else
            {
            this->m_exec_conf->msg->error() << "No patch enabled:" << quantity << " not registered." << std::endl;
            throw std::runtime_error("Error getting log value");
            }
        }
    else if (quantity == "mcm_patch_rcut")
        {
        if (m_patch)
            {
            return (Scalar)m_patch->getRCut();
            }
        else
            {
            this->m_exec_conf->msg->error() << "No patch enabled:" << quantity << " not registered." << std::endl;
            throw std::runtime_error("Error getting log value");
            }
        }
    else
        {
        //nothing found -> pass on to integrator
        return IntegratorMCM::getLogValue(quantity, timestep);
        }
    }

template <class Shape>
void IntegratorMCMMono<Shape>::printStats()
    {
    IntegratorMCM::printStats();

    /*unsigned int max_height = 0;
    unsigned int total_height = 0;

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int height = m_aabb_tree.height(i);
        if (height > max_height)
            max_height = height;
        total_height += height;
        }

    m_exec_conf->msg->notice(2) << "Avg AABB tree height: " << total_height / Scalar(m_pdata->getN()) << std::endl;
    m_exec_conf->msg->notice(2) << "Max AABB tree height: " << max_height << std::endl;*/
    }

template <class Shape>
bool IntegratorMCMMono<Shape>::attemptBoxResize(unsigned int timestep, const BoxDim& new_box)
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

template <class Shape>
void IntegratorMCMMono<Shape>::resetStats()
    {
    IntegratorMCM::resetStats();
    }

template <class Shape>
void IntegratorMCMMono<Shape>::slotNumTypesChange()
    {
    // call parent class method
    IntegratorMCM::slotNumTypesChange();

    // re-allocate the parameter storage
    m_params.resize(m_pdata->getNTypes());

    // skip the reallocation if the number of types does not change
    // this keeps old potential coefficients when restoring a snapshot
    // it will result in invalid coeficients if the snapshot has a different type id -> name mapping
    if (m_pdata->getNTypes() == m_overlap_idx.getW())
        return;

    // re-allocate overlap interaction matrix
    m_overlap_idx = Index2D(m_pdata->getNTypes());

    GPUArray<unsigned int> overlaps(m_overlap_idx.getNumElements(), m_exec_conf);
    m_overlaps.swap(overlaps);

    updateCellWidth();
    }

template <class Shape>
void IntegratorMCMMono<Shape>::update(unsigned int timestep)
    {

    if (max_density)
        {
        // IntegratorMCMMono<Shape>::writePairs();
        ;
        }

    else if (needs_mc && !max_density)
        {
        increment_mc_attempts();
        std::cout<<timestep<<" HPMC! "<<n_mc<<std::endl;

        // get needed vars
        ArrayHandle<mcm_counters_t> h_counters(m_count_total, access_location::host, access_mode::readwrite);
        mcm_counters_t& counters = h_counters.data[0];
        const BoxDim& box = m_pdata->getBox();
        unsigned int ndim = this->m_sysdef->getNDimensions();

        #ifdef ENABLE_MPI
        // compute the width of the active region
        Scalar3 npd = box.getNearestPlaneDistance();
        Scalar3 ghost_fraction = m_nominal_width / npd;
        #endif

        // Shuffle the order of particles for this n
        m_update_order.resize(m_pdata->getN());
        m_update_order.shuffle(timestep);

        const vec3<Scalar> defaultOrientation2D(0,1,0); //default long axis for 2D spherocylinders
        const vec3<Scalar> defaultOrientation3D(0,0,1); //default long axis for 3D spherocylinders
        unsigned int sweeps=10000;//m_pdata->getN();

        // update the AABB Tree
        buildAABBTree();
        // limit m_d entries so that particles cannot possibly wander more than one box image in one time n
        limitMoveDistances();
        // update the image list
        updateImageList();

        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "MCM update");

        if( m_external ) // I think we need this here otherwise I don't think it will get called.
            {
            m_external->compute(timestep);
            }

        // access interaction matrix
        ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

        // loop over local particles sweeps times

        for (unsigned int i_nselect = 0; i_nselect < sweeps; i_nselect++)
            {
            // access particle data and system box
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

            //access move sizes
            ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);

            // loop through N particles in a shuffled order
            for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
                {
                unsigned int i = m_update_order[cur_particle];

                // read in the current position and orientation
                Scalar4 postype_i = h_postype.data[i];
                Scalar4 orientation_i = h_orientation.data[i];
                vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

                #ifdef ENABLE_MPI
                if (m_comm)
                    {
                    // only move particle if active
                    if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                        continue;
                    }
                #endif

                // make a trial move for i
                hoomd::detail::Saru rng_i(i, m_seed + m_exec_conf->getRank()*m_nselect + i_nselect, timestep);
                unsigned int typ_i = __scalar_as_int(postype_i.w);
                Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
                bool move_type_translate = true;

                Shape shape_old(quat<Scalar>(orientation_i), m_params[typ_i]);

                // skip if no overlap check is required
                if (h_d.data[typ_i] == 0.0)
                    {
                    counters.translate_accept_count++;
                    continue;
                    }

                vec3<Scalar> or_vect_i(0,0,0);

                quat<Scalar> or_i=quat<Scalar>(orientation_i);
                if (ndim==2)
                    {
                    or_vect_i=rotate(or_i,defaultOrientation2D);
                    }
                else if (ndim==3)
                    {
                    or_vect_i=rotate(or_i,defaultOrientation3D);
                    }

                vec3<Scalar> dr(Scalar(0.0), Scalar(0.0), Scalar(0.0));
                Scalar d=h_d.data[typ_i];
                double mag=rng_i.s(-d, d);
                dr=mag*or_vect_i;
                // pos_i+=dr;


                pos_i += dr;

                #ifdef ENABLE_MPI
                if (m_comm)
                    {
                    // check if particle has moved into the ghost layer, and skip if it is
                    if (!isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                        continue;
                    }
                #endif



                bool overlap=false;
                OverlapReal r_cut_patch = 0;

                if (m_patch && !m_patch_log)
                    {
                    r_cut_patch = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);
                    }

                // subtract minimum AABB extent from search radius
                OverlapReal R_query = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0),
                    r_cut_patch-getMinCoreDiameter()/(OverlapReal)2.0);
                detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

                // patch + field interaction deltaU
                double patch_field_energy_diff = 0;

                // check for overlaps with neighboring particle's positions (also calculate the new energy)
                // All image boxes (including the primary)
                const unsigned int n_images = m_image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
                    detail::AABB aabb = aabb_i_local;
                    aabb.translate(pos_i_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // handle j==i situations
                                    if ( j != i )
                                        {
                                        // load the position and orientation of the j particle
                                        postype_j = h_postype.data[j];
                                        orientation_j = h_orientation.data[j];
                                        }
                                    else
                                        {
                                        if (cur_image == 0)
                                            {
                                            // in the first image, skip i == j
                                            continue;
                                            }
                                        else
                                            {
                                            // If this is particle i and we are in an outside image, use the translated position and orientation
                                            postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                            orientation_j = quat_to_scalar4(shape_i.orientation);
                                            }
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                                    counters.overlap_checks++;
                                    if (h_overlaps.data[m_overlap_idx(typ_i, typ_j)]
                                        && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                        && test_overlap(r_ij, shape_i, shape_j, counters.overlap_err_count))
                                        {
                                        overlap = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap)
                            break;
                        }  // end loop over AABB nodes

                    if (overlap)
                        break;
                    } // end loop over images

                // If no overlaps and Metropolis criterion is met, accept
                // trial move and update positions  and/or orientations.
                if (!overlap && rng_i.d() < slow::exp(patch_field_energy_diff))
                    {
                    // increment accept counter and assign new position
                    if (!shape_i.ignoreStatistics())
                        {
                        if (move_type_translate)
                            counters.translate_accept_count++;
                        else
                            counters.rotate_accept_count++;
                        }

                    // update the position of the particle in the tree for future updates
                    detail::AABB aabb = aabb_i_local;
                    aabb.translate(pos_i);
                    m_aabb_tree.update(i, aabb);

                    // update position of particle
                    h_postype.data[i] = make_scalar4(pos_i.x,pos_i.y,pos_i.z,postype_i.w);

                    if (shape_i.hasOrientation())
                        {
                        h_orientation.data[i] = quat_to_scalar4(shape_i.orientation);
                        }
                    }
                else
                    {
                    if (!shape_i.ignoreStatistics())
                        {
                        // increment reject counter
                        if (move_type_translate)
                            counters.translate_reject_count++;
                        else
                            counters.rotate_reject_count++;
                        }
                    }
                } // end loop over all particles
            } // end loop over nselect

            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            // wrap particles back into box
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                box.wrap(h_postype.data[i], h_image.data[i]);
                }
            }

        // perform the grid shift
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

            // precalculate the grid shift
            hoomd::detail::Saru rng(timestep, this->m_seed, 0xf4a3210e);
            Scalar3 shift = make_scalar3(0,0,0);
            shift.x = rng.s(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
            shift.y = rng.s(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
            if (this->m_sysdef->getNDimensions() == 3)
                {
                shift.z = rng.s(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
                }
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                // read in the current position and orientation
                Scalar4 postype_i = h_postype.data[i];
                vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
                r_i += vec3<Scalar>(shift);
                h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
                box.wrap(h_postype.data[i], h_image.data[i]);
                }
            this->m_pdata->translateOrigin(shift);
            }
        #endif

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

        // migrate and exchange particles
        communicate(true);

        // all particle have been moved, the aabb tree is now invalid
        m_aabb_tree_invalid = true;



        needs_mc=false;

        }

    else
        {
        m_exec_conf->msg->notice(10) << "MCMMono update: " << timestep << std::endl;
        IntegratorMCM::update(timestep);

        // get needed vars
        ArrayHandle<mcm_counters_t> h_counters(m_count_total, access_location::host, access_mode::readwrite);
        mcm_counters_t& counters = h_counters.data[0];
        const BoxDim& box = m_pdata->getBox();
        const unsigned int ndim = this->m_sysdef->getNDimensions();

        const BoxDim& curBox = m_pdata->getGlobalBox();

        const Scalar3& box_L = curBox.getL(); //save current box dimensions
        const double sep_tol=1.0001; //1.0001; //temp
        const int attempt_cutoff=10000;//m_pdata->getN(); //cutoff number of overlap removal attempts
        int n_attempts=0;  //counter for compression attempts
        const vec3<Scalar> defaultOrientation2D(0,1,0); //default long axis for 2D spherocylinders
        const vec3<Scalar> defaultOrientation3D(0,0,1); //default long axis for 3D spherocylinders
        const vec3<Scalar> x_norm(1,0,0);
        const vec3<Scalar> y_norm(0,1,0);
        double contact=0.001*box_L.x;

        double scale_factor=1-small; //factor to scale the box length by at each timestep, hardcoded for now, will add interface later
        // const double tol=1e-5;
        const double tiny=1e-7;
        // const double pi = 3.14159265358979323846;

        // //attempt to shrink box dimensions by scale_factor
        // Scalar3 L=make_scalar3(box_L.x*std::cbrt(scale_factor),box_L.y*std::cbrt(scale_factor),box_L.z*std::cbrt(scale_factor));
        // BoxDim newBox = m_pdata->getGlobalBox();
        // newBox.setL(L);
        // attemptBoxResize(timestep, newBox);

        #ifdef ENABLE_MPI
        // compute the width of the active region
        Scalar3 npd = box.getNearestPlaneDistance();
        Scalar3 ghost_fraction = m_nominal_width / npd;
        #endif

        // Shuffle the order of particles for this n
        m_update_order.resize(m_pdata->getN());
        m_update_order.shuffle(timestep);

        // update the AABB Tree
        buildAABBTree();
        // limit m_d entries so that particles cannot possibly wander more than one box image in one time n
        limitMoveDistances();
        // update the image list
        updateImageList();

        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "MCM update");

        if( m_external ) // I think we need this here otherwise I don't think it will get called.
            {
            m_external->compute(timestep);
            }

        // access interaction matrix
        ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

        // loop over all particles

        bool overlap=false;
        bool first=true;

        Scalar4 old_positions [m_pdata->getN()];
        Scalar4 old_orientations [m_pdata->getN()];
        avg_contacts=0;
        int particle_contacts_cond [m_pdata->getN()];
        int particle_contacts_any [m_pdata->getN()];
        do {
            overlap=false;
            n_attempts++;

             // Shuffle the order of particles for this n
            m_update_order.resize(m_pdata->getN());
            m_update_order.shuffle(timestep);

            // update the AABB Tree
            buildAABBTree();
            // limit m_d entries so that particles cannot possibly wander more than one box image in one time n
            limitMoveDistances();
            // update the image list
            updateImageList();

            // access interaction matrix
            ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

            vec3<Scalar> min_array [m_pdata->getN()];// = {}; //stores minimum overlap vectors for each particle
            Scalar4 positions [m_pdata->getN()];// = {}; //stores updated particle positions until loop over particles completes
            Scalar4 orientations [m_pdata->getN()];// = {}; //stores updated particle orientations until loop over particles completes

            // access particle data and system box
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

            if (first)
                {
                for (unsigned int q=0;q< m_pdata->getN(); q++)
                    {
                    Scalar4 postype_q=h_postype.data[q];
                    Scalar4 orientation_q=h_orientation.data[q];
                    vec3<Scalar> pos_q=vec3<Scalar>(postype_q);
                    unsigned int typ_q= __scalar_as_int(postype_q.w);

                    old_positions[q]=make_scalar4(pos_q.x,pos_q.y,pos_q.z,__int_as_scalar(typ_q));
                    old_orientations[q]=make_scalar4(orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w);
                    }

                //attempt to shrink box dimensions by scale_factor
                Scalar3 L=make_scalar3(box_L.x*std::cbrt(scale_factor),box_L.y*std::cbrt(scale_factor),box_L.z*std::cbrt(scale_factor));
                BoxDim newBox = m_pdata->getGlobalBox();
                newBox.setL(L);
                attemptBoxResize(timestep, newBox);

                first=false;
                }

            // loop through N particles in a shuffled order
            // for (unsigned int i = 0; i < m_pdata->getN(); i++)
            //     {
            for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
                {
                unsigned int i = m_update_order[cur_particle];
                particle_contacts_any[i]=0;
                particle_contacts_cond[i]=0;
                // unsigned int i = m_update_order[cur_particle];
                OverlapReal delta_min=10000; //tracks smallest overlap for current particle

                // read in the current position and orientation
                Scalar4 postype_i = h_postype.data[i];
                Scalar4 orientation_i = h_orientation.data[i];
                vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

                if (isnan(pos_i.x))
                    {
                    std::cout<<"ERROR!!!!!!!!!!!!!!!!!!!"<<" "<<timestep<<" "<<n_attempts<<i<<std::endl;
                    }

                OverlapReal ax=0.0; //components of vector to remove overlaps
                OverlapReal ay=0.0;
                OverlapReal az=0.0;
                OverlapReal ar1=0.0;
                OverlapReal ar2=0.0;

                unsigned int j_min=1;
                vec3<Scalar> r_ij_min;
                vec3<Scalar> pos_i_image_min;

                vec3<Scalar> or_vect_i(0,0,0);

                quat<Scalar> or_i=quat<Scalar>(orientation_i);
                if (ndim==2)
                    {
                    or_vect_i=rotate(or_i,defaultOrientation2D);
                    }
                else if (ndim==3)
                    {
                    or_vect_i=rotate(or_i,defaultOrientation3D);
                    }
                vec3<Scalar> x_norm_local=rotate(or_i,x_norm); //transform perpendicular axis into local particle frame
                vec3<Scalar> y_norm_local=rotate(or_i,y_norm);

                #ifdef ENABLE_MPI
                if (m_comm)
                    {
                    // only move particle if active
                    if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                        continue;
                    }
                #endif

                unsigned int typ_i = __scalar_as_int(postype_i.w);
                Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);

                double radius_i=m_params[typ_i].sweep_radius;

                double length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
                (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
                (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));
                // double volume=(pi*radius_i*radius_i*length_i)+((4/3)*pi*radius_i*radius_i*radius_i);
                // double moment_i=16.0*(length_i/radius_i)*(length_i*length_i)/12.0;
                double aspect_ratio=length_i/(2.0*radius_i);
                double moment_i=1.7+0.5*(aspect_ratio*aspect_ratio);
                if (moment_i==0.0)
                    {
                    moment_i+=1.0;
                    }

                std::vector<std::vector<Scalar>> tmp = getContactDistance(i);

                unsigned int neighborIDs[tmp[0].size()];
                Scalar siArr[tmp[0].size()];
                Scalar sjArr[tmp[0].size()];

                vec3<Scalar> images[tmp[0].size()];

                std::vector<Scalar>::iterator result = std::min_element(tmp[1].begin(), tmp[1].end());
                int min_index = std::distance(tmp[1].begin(), result);

                unsigned int min_id = tmp[0][min_index];
                Scalar rmin = tmp[1][min_index];


                for (int q=0;q<tmp[0].size();q++)
                    {
                    unsigned int ID = (unsigned int) tmp[0][q];
                    Scalar si1 = tmp[1][q];
                    Scalar sj1 = tmp[2][q];

                    double pos_i_image_x = tmp[3][q];
                    double pos_i_image_y = tmp[4][q];
                    double pos_i_image_z = tmp[5][q];

                    vec3<Scalar> pos_i_image_1(pos_i_image_x,pos_i_image_y,pos_i_image_z);

                    neighborIDs[q] = ID;
                    siArr[q] = si1;
                    sjArr[q] = sj1;
                    images[q] = pos_i_image_1;
                    }

                for (int q=0;q<tmp[0].size();q++)
                    {

                    vec3<Scalar> pos_i_image_tmp = images[q];
                    Scalar si_tmp = siArr[q];
                    Scalar sj_tmp = sjArr[q];
                    unsigned int j_tmp = neighborIDs[q];

                    Scalar4 orientation_j = h_orientation.data[j_tmp];
                    quat<Scalar> or_j=quat<Scalar>(orientation_j);
                    Scalar4 postype_j = h_postype.data[j_tmp];
                    vec3<Scalar> pos_j = vec3<Scalar>(postype_j);
                    or_j=quat<Scalar>(orientation_j);

                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                    double radius_j=m_params[typ_j].sweep_radius;

                    double length_j=sqrt((m_params[typ_j].x[0]-m_params[typ_j].x[1])*(m_params[typ_j].x[0]-m_params[typ_j].x[1])+
                                    (m_params[typ_j].y[0]-m_params[typ_j].y[1])*(m_params[typ_j].y[0]-m_params[typ_j].y[1])+
                                    (m_params[typ_j].z[0]-m_params[typ_j].z[1])*(m_params[typ_j].z[0]-m_params[typ_j].z[1]));

                    vec3<Scalar> or_vect_i;
                    vec3<Scalar> or_vect_j;
                    if (ndim==3)
                        {
                        or_vect_i=rotate(or_i,defaultOrientation3D);
                        or_vect_j=rotate(or_j,defaultOrientation3D);
                        }

                    else if (ndim==2)
                        {
                        or_vect_i=rotate(or_i,defaultOrientation2D);
                        or_vect_j=rotate(or_j,defaultOrientation2D);
                        }

                    vec3<Scalar> r_ij_tmp = vec3<Scalar>(postype_j) - pos_i_image_tmp;

                    vec3<Scalar> k_vect=(pos_i_image_tmp+si_tmp*or_vect_i)-(pos_j+sj_tmp*or_vect_j);

                    double mag_k=sqrt(dot(k_vect,k_vect));

                    double delta=(radius_i+radius_j)-mag_k;

                    if (delta>-contact && i!=j_tmp && typ_i==typ_j && typ_i==0)
                        {
                        avg_contacts++;
                        particle_contacts_cond[i]++;
                        }

                    else if (delta>-contact && i!=j_tmp)
                        {
                        particle_contacts_any[i]++;
                        }

                    if (delta>0 && i!=j_tmp) //particles are overlapping
                        {
                        overlap=true;
                        if (delta<delta_min)  //keep track of smallest overlap
                            {
                            delta_min=delta;
                            r_ij_min=r_ij_tmp;
                            pos_i_image_min=pos_i_image_tmp;
                            min_array[i]=k_vect;
                            j_min=j_tmp;
                            }

                        ax+=delta*k_vect.x/mag_k;
                        ay+=delta*k_vect.y/mag_k;
                        if (ndim==3)
                            {
                            az+=delta*k_vect.z/mag_k;

                            double k_ar1=dot(k_vect,x_norm_local); //project rotation axis onto k_vect
                            double k_ar2=dot(k_vect,y_norm_local);

                            ar1+=(delta*si_tmp/moment_i)*(k_ar1/mag_k);
                            ar2+=(delta*si_tmp/moment_i)*(k_ar2/mag_k);
                            }
                        else if (ndim==2)
                            {
                            az+=0;
                            ar1+=sep_tol*(delta*si_tmp/moment_i)/mag_k;
                            ar2+=0;
                            }
                        }
                    }


                // update position of particle in temporary copy

                vec3<Scalar> k_min=min_array[i];

                if (k_min!=vec3<Scalar>(0,0,0))
                    {
                    double a_mag=sqrt(ax*ax+ay*ay+az*az+ar1*ar1+ar2*ar2);

                    double goal_mag=(1.0-scale_factor)/16.0;
                    double scale_mag=goal_mag/a_mag;
                    ax*=scale_mag;
                    ay*=scale_mag;
                    az*=scale_mag;
                    ar1*=scale_mag;
                    ar2*=scale_mag;

                    vec3<Scalar> a(ax,ay,az); //center of mass displacement
                    vec3<Scalar> x_local;
                    vec3<Scalar> y_local;
                    Scalar4 or_x;
                    Scalar4 or_y;
                    double mag_x;
                    double mag_y;
                    quat<Scalar> quat_x;
                    quat<Scalar> quat_y;
                    quat<Scalar> quat_i;
                    quat<Scalar> new_quat_x;
                    quat<Scalar> new_quat_y;

                    if (ndim==3)
                        {
                        x_local=sin(ar1/2.0)*x_norm_local; //construct imaginary parts of quaternions
                        y_local=sin(ar2/2.0)*y_norm_local;

                        or_x=make_scalar4(cos(ar1/2.0),x_local.x,x_local.y,x_local.z); //add real parts
                        or_y=make_scalar4(cos(ar2/2.0),y_local.x,y_local.y,y_local.z);

                        quat<Scalar> new_quat_x(or_x);  //make quats
                        quat<Scalar> new_quat_y(or_y);

                        mag_x=sqrt(norm2(new_quat_x)); //normalize quats
                        mag_y=sqrt(norm2(new_quat_y));
                        quat_x=(1.0/mag_x)*new_quat_x;
                        quat_y=(1.0/mag_y)*new_quat_x;

                        or_i=quat_x*or_i;  //use quats to update orientation of particle i
                        or_i=quat_y*or_i;
                        }
                    else if (ndim==2)
                        {
                        or_x=make_scalar4(cos(ar1/2),0.0,0.0,sin(ar1/2));
                        quat<Scalar> quat_i(or_x);

                        mag_x=sqrt(norm2(quat_i)); //normalize quats
                        quat_x=(1.0/mag_x)*quat_i;

                        or_i=quat_x*or_i;  //use quats to update orientation of particle i
                        }

                    pos_i+=a;

                    // Scalar4 orientation_i = h_orientation.data[i];
                    Scalar4 postype_i = h_postype.data[i];
                    Scalar4 orientation_j = h_orientation.data[j_min];
                    Scalar4 postype_j = h_postype.data[j_min];
                    vec3<Scalar> pos_j = vec3<Scalar>(postype_j);
                    quat<Scalar> or_j=quat<Scalar>(orientation_j);

                    unsigned int typ_i = __scalar_as_int(postype_i.w);
                    double radius_i=m_params[typ_i].sweep_radius;
                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                    double radius_j=m_params[typ_j].sweep_radius;

                    double length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
                                    (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
                                    (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));

                    double length_j=sqrt((m_params[typ_j].x[0]-m_params[typ_j].x[1])*(m_params[typ_j].x[0]-m_params[typ_j].x[1])+
                                    (m_params[typ_j].y[0]-m_params[typ_j].y[1])*(m_params[typ_j].y[0]-m_params[typ_j].y[1])+
                                    (m_params[typ_j].z[0]-m_params[typ_j].z[1])*(m_params[typ_j].z[0]-m_params[typ_j].z[1]));

                    vec3<Scalar> or_vect_i;
                    vec3<Scalar> or_vect_j;
                    double si=1000;
                    double sj=1000;
                    if (ndim==3)
                        {
                        or_vect_i=rotate(or_i,defaultOrientation3D);
                        or_vect_j=rotate(or_j,defaultOrientation3D);

                        vec3<Scalar> N=cross(or_vect_i,or_vect_j);

                        double mag_N=sqrt(dot(N,N));
                        // vec3<Scalar> pos_vect=pos_j-pos_i;

                        if (mag_N==0) //parallel
                        // if (mag_N<tiny) //parallel
                            {
                            sj=0;
                            si=dot(or_vect_i,r_ij_min);
                            }
                        else
                            {
                            double intersect_test=abs(dot(N,r_ij_min));

                            if (intersect_test==0) //particle axis intersect
                            // if (intersect_test<tiny) //particle axis intersect
                                {
                                Eigen::MatrixXf a(3, 2);
                                Eigen::MatrixXf b(1, 3);
                                Eigen::MatrixXf x(1, 2);

                                double a00=or_vect_i.x;
                                double a01=or_vect_j.x;
                                double a10=or_vect_i.y;
                                double a11=or_vect_j.y;
                                double a20=or_vect_i.z;
                                double a21=or_vect_j.z;

                                double b00=-r_ij_min.x;
                                double b10=-r_ij_min.y;
                                double b20=-r_ij_min.z;

                                a(0,0)=a00;
                                a(0,1)=a01;
                                a(1,0)=a10;
                                a(1,1)=a11;
                                a(2,0)=a20;
                                a(2,1)=a21;

                                b(0,0)=b00;
                                b(1,0)=b10;
                                b(2,0)=b20;

                                Eigen::Matrix2f c(2,2);
                                c=a.transpose() * a;
                                Eigen::Matrix2f d(2,2);
                                d=c.inverse();
                                Eigen::MatrixXf e(2,3);
                                e=d * a.transpose();
                                x=e * b;

                                si=x(0,0);
                                sj=x(1,0);
                                }
                            else //skew axis, calculate closest approach
                                {
                                vec3<Scalar> n1=cross(or_vect_i,N);
                                vec3<Scalar> n2=cross(or_vect_j,N);

                                si=dot(r_ij_min,n2)/dot(or_vect_i,n2);
                                sj=dot(-r_ij_min,n1)/dot(or_vect_j,n1);
                                }
                            }
                        }
                    else if (ndim==2)
                        {
                        or_vect_i=rotate(or_i,defaultOrientation2D);
                        or_vect_j=rotate(or_j,defaultOrientation2D);

                        vec3<Scalar> N=cross(or_vect_i,or_vect_j);
                        double mag_N=sqrt(dot(N,N));

                        if (mag_N<tiny) //parallel
                            {
                            sj=0;
                            // vec3<Scalar> pos_vect=pos_j-pos_i;
                            si=dot(or_vect_i,r_ij_min);
                            }

                        else
                            {
                            // vec3<Scalar> pos_vect=pos_j-pos_i;
                            si=(r_ij_min.x-(r_ij_min.y/or_vect_j.y))/(or_vect_i.x-(or_vect_i.y/or_vect_j.y));
                            sj=(r_ij_min.y-si*or_vect_i.y)/or_vect_j.y;
                            }
                        }

                    //truncate nearest approach search to lengths of particles
                    if (si>length_i/2.0)
                        {
                        si=length_i/2.0;
                        }
                    else if (si<-length_i/2.0)
                        {
                        si=-length_i/2.0;
                        }
                    if (sj>length_j/2.0)
                        {
                        sj=length_j/2.0;
                        }
                    else if (sj<-length_j/2.0)
                        {
                        sj=-length_j/2.0;
                        }

                    vec3<Scalar> k_vect=(pos_i_image_min+si*or_vect_i)-(pos_j+sj*or_vect_j);

                    double mag_k=sqrt(dot(k_vect,k_vect));
                    double delta_new=(radius_i+radius_j)-mag_k;

                    double diff_delta=delta_min-delta_new;
                    // double disp_mag=sep_tol;
                    double disp_mag;
                    if (diff_delta==0)
                        {
                        disp_mag=0;
                        }
                    else
                        {
                        disp_mag=(((sep_tol*(delta_min/2.0))/diff_delta)+1.0)*diff_delta;
                        }

                    a=disp_mag*a;
                    ar1=ar1*disp_mag;
                    ar2=ar2*disp_mag;

                    if (ndim==3)
                        {
                        x_local=sin(ar1/2.0)*x_norm_local; //construct imaginary parts of quaternions
                        y_local=sin(ar2/2.0)*y_norm_local;

                        or_x=make_scalar4(cos(ar1/2.0),x_local.x,x_local.y,x_local.z); //add real parts
                        or_y=make_scalar4(cos(ar2/2.0),y_local.x,y_local.y,y_local.z);

                        quat<Scalar> revised_quat_x(or_x);  //make quats
                        quat<Scalar> revised_quat_y(or_y);

                        mag_x=sqrt(norm2(revised_quat_x)); //normalize quats
                        mag_y=sqrt(norm2(revised_quat_y));
                        quat_x=(1.0/mag_x)*revised_quat_x;
                        quat_y=(1.0/mag_y)*revised_quat_x;

                        or_i=quat_x*or_i;  //use quats to update orientation of particle i
                        or_i=quat_y*or_i;
                        }
                    else if (ndim==2)
                        {
                        or_x=make_scalar4(cos(ar1/2),0.0,0.0,sin(ar1/2));
                        quat<Scalar> revised_quat_i(or_x);

                        mag_x=sqrt(norm2(quat_i)); //normalize quats
                        quat<Scalar> new_quat_x=(1.0/mag_x)*quat_i;

                        or_i=new_quat_x*or_i;  //use quats to update orientation of particle i
                        }
                    // or_vect_i=rotate(or_i,defaultOrientation3D);
                    // or_vect_j=rotate(or_j,defaultOrientation3D);

                    pos_i+=a;
                    }
                    positions[i] = make_scalar4(pos_i.x,pos_i.y,pos_i.z,postype_i.w);
                    orientations[i] = quat_to_scalar4(or_i);  //store in copy in correct format
                } // end loop over all particles
            avg_contacts=avg_contacts/m_pdata->getN();
            avg_contacts=avg_contacts/2; //2 to avoid double counting contacts, each pair indexed twice

            for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
                {
                unsigned int k = cur_particle;

                Scalar4 postype_copy_k = positions[k];

                Scalar4 orientation_copy_k = orientations[k];
                vec3<Scalar> pos_copy_k = vec3<Scalar>(postype_copy_k);

                int typ_copy_k = __scalar_as_int(postype_copy_k.w);
                Shape shape_copy_k(quat<Scalar>(orientation_copy_k), m_params[typ_copy_k]);

                OverlapReal r_cut_patch = 0;

                if (m_patch && !m_patch_log)
                    {
                    r_cut_patch = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_copy_k);
                    }
                OverlapReal R_query_k = std::max(shape_copy_k.getCircumsphereDiameter()/OverlapReal(2.0),
                r_cut_patch-getMinCoreDiameter()/(OverlapReal)2.0);
                detail::AABB aabb_k_local = detail::AABB(vec3<Scalar>(0,0,0),R_query_k);

                // check for overlaps with neighboring particle's positions (also calculate the new energy)
                // All image boxes (including the primary)
                const unsigned int n_images = m_image_list.size();

                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_k_image = pos_copy_k + m_image_list[cur_image];
                    detail::AABB aabb = aabb_k_local;
                    aabb.translate(pos_k_image);
                    } // end loop over images

                // update the position of the particle in the tree for future updates
                detail::AABB aabb_k = aabb_k_local;
                aabb_k.translate(pos_copy_k);
                m_aabb_tree.update(k, aabb_k);

                h_postype.data[k] = make_scalar4(pos_copy_k.x,pos_copy_k.y,pos_copy_k.z,postype_copy_k.w); //update position of particle

                if (shape_copy_k.hasOrientation())
                    {
                    h_orientation.data[k] = quat_to_scalar4(shape_copy_k.orientation);
                    }
                }
            } while(!done && overlap && n_attempts<=attempt_cutoff); //end overlap while loop
            std::cout<<n_attempts<<std::endl;

            if (n_attempts>attempt_cutoff)
                {
                // max_density=true;
                // if (n_mc<mc_cutoff)
                //     {
                //     needs_mc=true; //use monte carlo to anneal
                //     n_attempts=0;
                //     }

                // std::ofstream outfile3;

                // outfile3.open("Contacts.txt", std::ios_base::app);
                // outfile3<<"-1"<<" "<<"-1"<<std::endl;
                // for (int ii=0;ii<m_pdata->getN();ii++)
                //     {
                //     outfile3<<particle_contacts_cond[ii]<<" "<<particle_contacts_any[ii]<<" "<<std::endl;
                //     }

                // outfile3.close();
                if (small>1e-3)
                    {
                    Scalar3 L=make_scalar3(box_L.x/std::cbrt(scale_factor),box_L.y/std::cbrt(scale_factor),box_L.z/std::cbrt(scale_factor));  //attempt to shrink box dimensions by scale_factor
                    BoxDim newBox = m_pdata->getGlobalBox();
                    newBox.setL(L);
                    attemptBoxResize(timestep, newBox);

                    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);

                    for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
                        {
                        unsigned int k = cur_particle;

                        Scalar4 postype_copy_k = old_positions[k];

                        Scalar4 orientation_copy_k = old_orientations[k];
                        vec3<Scalar> pos_copy_k = vec3<Scalar>(postype_copy_k);

                        int typ_copy_k = __scalar_as_int(postype_copy_k.w);
                        Shape shape_copy_k(quat<Scalar>(orientation_copy_k), m_params[typ_copy_k]);

                        OverlapReal r_cut_patch = 0;

                        if (m_patch && !m_patch_log)
                            {
                            r_cut_patch = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_copy_k);
                            }
                        OverlapReal R_query_k = std::max(shape_copy_k.getCircumsphereDiameter()/OverlapReal(2.0),
                        r_cut_patch-getMinCoreDiameter()/(OverlapReal)2.0);
                        detail::AABB aabb_k_local = detail::AABB(vec3<Scalar>(0,0,0),R_query_k);

                        // check for overlaps with neighboring particle's positions (also calculate the new energy)
                        // All image boxes (including the primary)
                        unsigned int n_images = m_image_list.size();
                        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                            {
                            vec3<Scalar> pos_k_image = pos_copy_k + m_image_list[cur_image];
                            detail::AABB aabb = aabb_k_local;
                            aabb.translate(pos_k_image);
                            } // end loop over images

                        // update the position of the particle in the tree for future updates
                        detail::AABB aabb_k = aabb_k_local;
                        aabb_k.translate(pos_copy_k);
                        m_aabb_tree.update(k, aabb_k);

                        h_postype.data[k] = make_scalar4(pos_copy_k.x,pos_copy_k.y,pos_copy_k.z,postype_copy_k.w); //update position of particle

                        if (shape_copy_k.hasOrientation())
                            {
                            h_orientation.data[k] = quat_to_scalar4(shape_copy_k.orientation);
                            }
                        }

                    small=small*0.1;
                    // needs_mc=true;
                    n_attempts=0;
                    }
                else
                    {
                    std::cout<<"Test1"<<std::endl;
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.0);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.1);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.2);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.3);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.4);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.5);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.6);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.7);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.8);
                    IntegratorMCMMono<Shape>::diffuseConductivity(0.9);
                    std::cout<<"Test2"<<std::endl;
                    IntegratorMCMMono<Shape>::writePairs(0.0);
                    IntegratorMCMMono<Shape>::writePairs(0.1);
                    IntegratorMCMMono<Shape>::writePairs(0.2);
                    IntegratorMCMMono<Shape>::writePairs(0.3);
                    IntegratorMCMMono<Shape>::writePairs(0.4);
                    IntegratorMCMMono<Shape>::writePairs(0.5);
                    IntegratorMCMMono<Shape>::writePairs(0.6);
                    IntegratorMCMMono<Shape>::writePairs(0.7);
                    IntegratorMCMMono<Shape>::writePairs(0.8);
                    IntegratorMCMMono<Shape>::writePairs(0.9);
                    std::cout<<"Test3"<<std::endl;
                    max_density=true; //system is fully compressed
                    }
                }
            else
                {
                n_attempts=0; //error catching
                }

            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            // wrap particles back into box
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                box.wrap(h_postype.data[i], h_image.data[i]);
                }
            }

        // perform the grid shift
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

            // precalculate the grid shift
            hoomd::detail::Saru rng(timestep, this->m_seed, 0xf4a3210e);
            Scalar3 shift = make_scalar3(0,0,0);
            shift.x = rng.s(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
            shift.y = rng.s(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
            if (this->m_sysdef->getNDimensions() == 3)
                {
                shift.z = rng.s(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
                }
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                // read in the current position and orientation
                Scalar4 postype_i = h_postype.data[i];
                vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
                r_i += vec3<Scalar>(shift);
                h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
                box.wrap(h_postype.data[i], h_image.data[i]);
                }
            this->m_pdata->translateOrigin(shift);
            }
        #endif

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

        // migrate and exchange particles
        communicate(true);

        // all particle have been moved, the aabb tree is now invalid
        m_aabb_tree_invalid = true;
        }

        /*! \param timestep current n
            \param early_exit exit at first overlap found if true
            \returns number of overlaps if early_exit=false, 1 if early_exit=true
        */
        } //end max_density check

template <class Shape>
std::vector<std::vector<Scalar>> IntegratorMCMMono<Shape>::getContactDistance(unsigned int i)
    {
    // get needed vars
    const BoxDim& box = m_pdata->getBox();
    const unsigned int ndim = this->m_sysdef->getNDimensions();
    const BoxDim& curBox = m_pdata->getGlobalBox();

    const Scalar3& box_L = curBox.getL(); //save current box dimensions
    const vec3<Scalar> defaultOrientation2D(0,1,0); //default long axis for 2D spherocylinders
    const vec3<Scalar> defaultOrientation3D(0,0,1); //default long axis for 3D spherocylinders
    const vec3<Scalar> x_norm(1,0,0);
    const vec3<Scalar> y_norm(0,1,0);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);

    std::vector<Scalar> nearNeighbors;
    std::vector<Scalar> imagesx;
    std::vector<Scalar> imagesy;
    std::vector<Scalar> imagesz;
    std::vector<Scalar> iaxis;
    std::vector<Scalar> jaxis;
    // std::vector<Scalar> typj;

    std::vector<std::vector<Scalar>> neighborStats;

    // update the AABB Tree
    buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one time n
    limitMoveDistances();
    // update the image list
    updateImageList();


    // read in the current position and orientation
    Scalar4 postype_i = h_postype.data[i];
    Scalar4 orientation_i = h_orientation.data[i];
    vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

    double mag_k = -999;

    OverlapReal ax=0.0; //components of vector to remove overlaps
    OverlapReal ay=0.0;
    OverlapReal az=0.0;
    OverlapReal ar1=0.0;
    OverlapReal ar2=0.0;

    unsigned int j_min=1;
    vec3<Scalar> r_ij_min;
    vec3<Scalar> pos_i_image_min;

    vec3<Scalar> or_vect_i(0,0,0);

    quat<Scalar> or_i=quat<Scalar>(orientation_i);
    if (ndim==2)
        {
        or_vect_i=rotate(or_i,defaultOrientation2D);
        }
    else if (ndim==3)
        {
        or_vect_i=rotate(or_i,defaultOrientation3D);
        }
    vec3<Scalar> x_norm_local=rotate(or_i,x_norm); //transform perpendicular axis into local particle frame
    vec3<Scalar> y_norm_local=rotate(or_i,y_norm);

    unsigned int typ_i = __scalar_as_int(postype_i.w);
    Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);

    double radius_i=m_params[typ_i].sweep_radius;

    double length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
    (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
    (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));

    OverlapReal r_cut_patch = 0;

    if (m_patch && !m_patch_log)
        {
        r_cut_patch = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);
        }

    // subtract minimum AABB extent from search radius
    OverlapReal R_query_i = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0),
        r_cut_patch-getMinCoreDiameter()/(OverlapReal)2.0);
    detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query_i);

    // check for overlaps with neighboring particle's positions (also calculate the new energy)
    // All image boxes (including the primary)
    const unsigned int n_images = m_image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
        detail::AABB aabb = aabb_i_local;
        aabb.translate(pos_i_image);

        // stackless search
        for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
            {
            if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                {
                if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        // read in its position and orientation
                        unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        Scalar4 postype_j;
                        Scalar4 orientation_j;

                        // handle j==i situations
                        if ( j != i )
                            {
                            // load the position and orientation of the j particle
                            postype_j = h_postype.data[j];
                            orientation_j = h_orientation.data[j];
                            }
                        else
                            {
                            if (cur_image == 0)
                                {
                                // in the first image, skip i == j
                                continue;
                                }
                            else
                                {
                                // If this is particle i and we are in an outside image, use the translated position and orientation
                                postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                orientation_j = quat_to_scalar4(shape_i.orientation);
                                }
                            }
                        // postype_j = h_postype.data[j];
                        // orientation_j = h_orientation.data[j];

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);
                        quat<Scalar> or_j=quat<Scalar>(orientation_j);

                        double radius_j=m_params[typ_j].sweep_radius;

                        // counters.overlap_checks++;

                        vec3<Scalar> pos_j = vec3<Scalar>(postype_j);

                        double length_j=sqrt((m_params[typ_j].x[0]-m_params[typ_j].x[1])*(m_params[typ_j].x[0]-m_params[typ_j].x[1])+
                        (m_params[typ_j].y[0]-m_params[typ_j].y[1])*(m_params[typ_j].y[0]-m_params[typ_j].y[1])+
                        (m_params[typ_j].z[0]-m_params[typ_j].z[1])*(m_params[typ_j].z[0]-m_params[typ_j].z[1]));

                        double centerDist2=sqrt(dot(r_ij,r_ij));

                        //Sanity check, particles cannot be in contact if condition is met
                        if (centerDist2>(length_i+length_j+radius_i+radius_j))
                            {
                            continue;
                            }

                        //return vectors along spherocylinder axis
                        vec3<Scalar> or_vect_i;
                        vec3<Scalar> or_vect_j;

                        double si=1000;
                        double sj=1000;
                        if (ndim==3)
                            {
                            or_vect_i=rotate(or_i,defaultOrientation3D);
                            or_vect_j=rotate(or_j,defaultOrientation3D);

                            vec3<Scalar> N=cross(or_vect_i,or_vect_j);

                            double mag_N=sqrt(dot(N,N));

                            if (mag_N==0) //parallel
                                {
                                sj=0;
                                si=dot(or_vect_i,r_ij);
                                }
                            else
                                {
                                double intersect_test=abs(dot(N,r_ij));

                                if (intersect_test==0) //particle axis intersect
                                    {
                                    Eigen::MatrixXf a(3, 2);
                                    Eigen::MatrixXf b(1, 3);
                                    Eigen::MatrixXf x(1, 2);

                                    double a00=or_vect_i.x;
                                    double a01=or_vect_j.x;
                                    double a10=or_vect_i.y;
                                    double a11=or_vect_j.y;
                                    double a20=or_vect_i.z;
                                    double a21=or_vect_j.z;

                                    double b00=-r_ij.x;
                                    double b10=-r_ij.y;
                                    double b20=-r_ij.z;

                                    a(0,0)=a00;
                                    a(0,1)=a01;
                                    a(1,0)=a10;
                                    a(1,1)=a11;
                                    a(2,0)=a20;
                                    a(2,1)=a21;

                                    b(0,0)=b00;
                                    b(1,0)=b10;
                                    b(2,0)=b20;

                                    Eigen::Matrix2f c(2,2);
                                    c=a.transpose() * a;
                                    Eigen::Matrix2f d(2,2);
                                    d=c.inverse();
                                    Eigen::MatrixXf e(2,3);
                                    e=d * a.transpose();
                                    x=e * b;

                                    si=x(0,0);
                                    sj=x(1,0);
                                    }
                                else //skew axis, calculate closest approach
                                    {
                                    vec3<Scalar> n1=cross(or_vect_i,N);
                                    vec3<Scalar> n2=cross(or_vect_j,N);

                                    si=dot(r_ij,n2)/dot(or_vect_i,n2);
                                    sj=dot(-r_ij,n1)/dot(or_vect_j,n1);
                                    }
                                }
                            }
                        else if (ndim==2)
                            {
                            or_vect_i=rotate(or_i,defaultOrientation2D);
                            or_vect_j=rotate(or_j,defaultOrientation2D);

                            vec3<Scalar> N=cross(or_vect_i,or_vect_j);
                            double mag_N=sqrt(dot(N,N));

                            if (mag_N<0) //parallel
                                {
                                sj=0;
                                si=dot(or_vect_i,r_ij);
                                }

                            else
                                {
                                si=(r_ij.x-(r_ij.y/or_vect_j.y))/(or_vect_i.x-(or_vect_i.y/or_vect_j.y));
                                sj=(r_ij.y-si*or_vect_i.y)/or_vect_j.y;
                                }
                            }

                        //truncate nearest approach search to lengths of particles
                        if (si>length_i/2.0)
                            {
                            si=length_i/2.0;
                            }
                        else if (si<-length_i/2.0)
                            {
                            si=-length_i/2.0;
                            }
                        if (sj>length_j/2.0)
                            {
                            sj=length_j/2.0;
                            }
                        else if (sj<-length_j/2.0)
                            {
                            sj=-length_j/2.0;
                            }

                        vec3<Scalar> k_vect=(pos_i_image+si*or_vect_i)-(pos_j+sj*or_vect_j);

                        nearNeighbors.push_back(j);
                        iaxis.push_back(si);
                        jaxis.push_back(sj);
                        imagesx.push_back(pos_i_image.x);
                        imagesy.push_back(pos_i_image.y);
                        imagesz.push_back(pos_i_image.z);
                        // typj.push_back(typ_j)
                        }
                    }
                }
            else
                {
                // skip ahead
                cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                }
            }  // end loop over AABB nodes
        } // end loop over images

    neighborStats.push_back(nearNeighbors);
    neighborStats.push_back(iaxis);
    neighborStats.push_back(jaxis);
    neighborStats.push_back(imagesx);
    neighborStats.push_back(imagesy);
    neighborStats.push_back(imagesz);
    // neighborStats.push_back(typj);

    return neighborStats;

    }

template <class Shape>
unsigned int IntegratorMCMMono<Shape>::countOverlaps(unsigned int timestep, bool early_exit)
    {
    unsigned int overlap_count = 0;
    unsigned int err_count = 0;

    m_exec_conf->msg->notice(10) << "MCMMono count overlaps: " << timestep << std::endl;

    if (!m_past_first_run)
        {
        m_exec_conf->msg->error() << "count_overlaps only works after a run() command" << std::endl;
        throw std::runtime_error("Error communicating in count_overlaps");
        }

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "MCM count overlaps");

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access parameters and interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // Loop over all particles
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // Check particle against AABB tree for neighbors
        detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                            if (h_tag.data[i] <= h_tag.data[j]
                                && h_overlaps.data[m_overlap_idx(typ_i,typ_j)]
                                && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                && test_overlap(r_ij, shape_i, shape_j, err_count)
                                && test_overlap(-r_ij, shape_j, shape_i, err_count))
                                {
                                overlap_count++;
                                if (early_exit)
                                    {
                                    // exit early from loop over neighbor particles
                                    break;
                                    }
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                if (overlap_count && early_exit)
                    {
                    break;
                    }
                } // end loop over AABB nodes

            if (overlap_count && early_exit)
                {
                break;
                }
            } // end loop over images

        if (overlap_count && early_exit)
            {
            break;
            }
        } // end loop over particles

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &overlap_count, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        if (early_exit && overlap_count > 1)
            overlap_count = 1;
        }
    #endif

    return overlap_count;
    }

template<class Shape>
float IntegratorMCMMono<Shape>::computePatchEnergy(unsigned int timestep)
    {
    // sum up in double precision
    double energy = 0.0;

    // return if nothing to do
    if (!m_patch) return energy;

    m_exec_conf->msg->notice(10) << "MCM compute patch energy: " << timestep << std::endl;

    if (!m_past_first_run)
        {
        m_exec_conf->msg->error() << "get_patch_energy only works after a run() command" << std::endl;
        throw std::runtime_error("Error communicating in count_overlaps");
        }

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "MCM compute patch energy");

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access parameters and interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // Loop over all particles
    #ifdef ENABLE_TBB
    energy = tbb::parallel_reduce(tbb::blocked_range<unsigned int>(0, m_pdata->getN()),
        0.0f,
        [&](const tbb::blocked_range<unsigned int>& r, float energy)->float {
        for (unsigned int i = r.begin(); i != r.end(); ++i)
    #else
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
    #endif
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        Scalar d_i = h_diameter.data[i];
        Scalar charge_i = h_charge.data[i];

        // the cut-off
        float r_cut = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);

        // subtract minimum AABB extent from search radius
        OverlapReal R_query = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0),
            r_cut-getMinCoreDiameter()/(OverlapReal)2.0);
        detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];
                            Scalar d_j = h_diameter.data[j];
                            Scalar charge_j = h_charge.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                            // count unique pairs within range
                            Scalar rcut_ij = r_cut + 0.5*m_patch->getAdditiveCutoff(typ_j);

                            if (h_tag.data[i] <= h_tag.data[j] && dot(r_ij,r_ij) <= rcut_ij*rcut_ij)
                                {
                                energy += m_patch->energy(r_ij,
                                       typ_i,
                                       quat<float>(orientation_i),
                                       d_i,
                                       charge_i,
                                       typ_j,
                                       quat<float>(orientation_j),
                                       d_j,
                                       charge_j);
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                } // end loop over AABB nodes
            } // end loop over images
        } // end loop over particles
    #ifdef ENABLE_TBB
    return energy;
    }, [](float x, float y)->float { return x+y; } );
    #endif

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    return energy;
    }


template <class Shape>
Scalar IntegratorMCMMono<Shape>::getMaxCoreDiameter()
    {
    // for each type, create a temporary shape and return the maximum diameter
    OverlapReal maxD = OverlapReal(0.0);
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        Shape temp(quat<Scalar>(), m_params[typ]);
        maxD = std::max(maxD, temp.getCircumsphereDiameter());
        }

    return maxD;
    }

template <class Shape>
OverlapReal IntegratorMCMMono<Shape>::getMinCoreDiameter()
    {
    // for each type, create a temporary shape and return the minimum diameter
    OverlapReal minD = OverlapReal(0.0);
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        Shape temp(quat<Scalar>(), m_params[typ]);
        minD = std::min(minD, temp.getCircumsphereDiameter());
        }

    if (m_patch)
        {
        OverlapReal max_extent = 0.0;
        for (unsigned int typ =0; typ < this->m_pdata->getNTypes(); typ++)
            max_extent = std::max(max_extent, (OverlapReal) m_patch->getAdditiveCutoff(typ));
        minD = std::max((OverlapReal) 0.0, minD-max_extent);
        }

    return minD;
    }

template <class Shape>
void IntegratorMCMMono<Shape>::setParam(unsigned int typ,  const param_type& param)
    {
    // validate input
    if (typ >= this->m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "integrate.mode_mcm_?." << /*evaluator::getName() <<*/ ": Trying to set pair params for a non existant type! "
                  << typ << std::endl;
        throw std::runtime_error("Error setting parameters in IntegratorMCMMono");
        }

    // need to scope this because updateCellWidth will access it
        {
        // update the parameter for this type
        m_exec_conf->msg->notice(7) << "setParam : " << typ << std::endl;
        m_params[typ] = param;
        }

    updateCellWidth();
    }

template <class Shape>
void IntegratorMCMMono<Shape>::setOverlapChecks(unsigned int typi, unsigned int typj, bool check_overlaps)
    {
    // validate input
    if (typi >= this->m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "integrate.mode_mcm_?." << /*evaluator::getName() <<*/ ": Trying to set interaction matrix for a non existant type! "
                  << typi << std::endl;
        throw std::runtime_error("Error setting interaction matrix in IntegratorMCMMono");
        }

    if (typj >= this->m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "integrate.mode_mcm_?." << /*evaluator::getName() <<*/ ": Trying to set interaction matrix for a non existant type! "
                  << typj << std::endl;
        throw std::runtime_error("Error setting interaction matrix in IntegratorMCMMono");
        }

    // update the parameter for this type
    m_exec_conf->msg->notice(7) << "setOverlapChecks : " << typi << " " << typj << " " << check_overlaps << std::endl;
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::readwrite);
    h_overlaps.data[m_overlap_idx(typi,typj)] = check_overlaps;
    h_overlaps.data[m_overlap_idx(typj,typi)] = check_overlaps;
    }

//! Calculate a list of box images within interaction range of the simulation box, innermost first
template <class Shape>
inline const std::vector<vec3<Scalar> >& IntegratorMCMMono<Shape>::updateImageList()
    {
    // cancel if the image list is up to date
    if (m_image_list_valid)
        return m_image_list;

    // triclinic boxes have 4 linearly independent body diagonals
    // box_circumsphere = max(body_diagonals)
    // range = getMaxCoreDiameter() + box_circumsphere
    // while still adding images, examine successively larger blocks of images, checking the outermost against range

    if (m_prof) m_prof->push(m_exec_conf, "MCM image list");

    unsigned int ndim = m_sysdef->getNDimensions();

    m_image_list_valid = true;
    m_image_list_is_initialized = true;
    m_image_list.clear();
    m_image_hkl.clear();
    m_image_list_rebuilds++;

    // Get box vectors
    const BoxDim& box = m_pdata->getGlobalBox();
    vec3<Scalar> e1 = vec3<Scalar>(box.getLatticeVector(0));
    vec3<Scalar> e2 = vec3<Scalar>(box.getLatticeVector(1));
    // 2D simulations don't necessarily have a zero-size z-dimension, but it is convenient for us if we assume one.
    vec3<Scalar> e3(0,0,0);
    if (ndim == 3)
        e3 = vec3<Scalar>(box.getLatticeVector(2));

    // Maximum interaction range is the sum of the system box circumsphere diameter and the max particle circumsphere diameter and move distance
    Scalar range = 0.0f;
    // Try four linearly independent body diagonals and find the longest
    vec3<Scalar> body_diagonal;
    body_diagonal = e1 - e2 - e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 - e2 + e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 + e2 - e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 + e2 + e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    range = fast::sqrt(range);

    Scalar max_trans_d_and_diam(0.0);
        {
        // access the type parameters
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

       // for each type, create a temporary shape and return the maximum sum of diameter and move size
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            Shape temp(quat<Scalar>(), m_params[typ]);

            Scalar r_cut_patch(0.0);
            if (m_patch)
                {
                r_cut_patch = (Scalar)m_patch->getRCut() + m_patch->getAdditiveCutoff(typ);
                }

            Scalar range_i = detail::max((Scalar)temp.getCircumsphereDiameter(),r_cut_patch);
            max_trans_d_and_diam = detail::max(max_trans_d_and_diam, range_i+Scalar(m_nselect)*h_d.data[typ]);
            }
        }

    range += max_trans_d_and_diam;

    // add any extra requested width
    range += m_extra_image_width;

    Scalar range_sq = range*range;

    // initialize loop
    int3 hkl;
    bool added_images = true;
    int hkl_max = 0;
    const int crazybig = 30;
    while (added_images == true)
        {
        added_images = false;

        int x_max = hkl_max;
        int y_max = hkl_max;
        int z_max = 0;
        if (ndim == 3)
            z_max = hkl_max;

        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
            if (di.getW() > 1) x_max = 0;
            if (di.getH() > 1) y_max = 0;
            if (di.getD() > 1) z_max = 0;
            }
        #endif

        // for h in -hkl_max..hkl_max
        //  for k in -hkl_max..hkl_max
        //   for l in -hkl_max..hkl_max
        //    check if exterior to box of images: if abs(h) == hkl_max || abs(k) == hkl_max || abs(l) == hkl_max
        //     if abs(h*e1 + k*e2 + l*e3) <= range; then image_list.push_back(hkl) && added_cells = true;
        for (hkl.x = -x_max; hkl.x <= x_max; hkl.x++)
            {
            for (hkl.y = -y_max; hkl.y <= y_max; hkl.y++)
                {
                for (hkl.z = -z_max; hkl.z <= z_max; hkl.z++)
                    {
                    // Note that the logic of the following line needs to work in 2 and 3 dimensions
                    if (abs(hkl.x) == hkl_max || abs(hkl.y) == hkl_max || abs(hkl.z) == hkl_max)
                        {
                        vec3<Scalar> r = Scalar(hkl.x) * e1 + Scalar(hkl.y) * e2 + Scalar(hkl.z) * e3;
                        // include primary image so we can do checks in in one loop
                        if (dot(r,r) <= range_sq)
                            {
                            vec3<Scalar> img = (Scalar)hkl.x*e1+(Scalar)hkl.y*e2+(Scalar)hkl.z*e3;
                            m_image_list.push_back(img);
                            m_image_hkl.push_back(make_int3(hkl.x, hkl.y, hkl.z));
                            added_images = true;
                            }
                        }
                    }
                }
            }
        if (!m_hkl_max_warning_issued && hkl_max > crazybig)
            {
            m_hkl_max_warning_issued = true;
            m_exec_conf->msg->warning() << "Exceeded sanity limit for image list, generated out to " << hkl_max
                                     << " lattice vectors. Logic error?" << std::endl
                                     << "This message will not be repeated." << std::endl;

            break;
            }

        hkl_max++;
        }

    // cout << "built image list" << std::endl;
    // for (unsigned int i = 0; i < m_image_list.size(); i++)
    //     cout << m_image_list[i].x << " " << m_image_list[i].y << " " << m_image_list[i].z << std::endl;
    // cout << std::endl;

    // warn the user if more than one image in each direction is activated
    unsigned int img_warning = 9;
    if (ndim == 3)
        {
        img_warning = 27;
        }
    if (!m_image_list_warning_issued && m_image_list.size() > img_warning)
        {
        m_image_list_warning_issued = true;
        m_exec_conf->msg->warning() << "Box size is too small or move size is too large for the minimum image convention." << std::endl
                                    << "Testing " << m_image_list.size() << " images per trial move, performance may slow." << std::endl
                                    << "This message will not be repeated." << std::endl;
        }

    m_exec_conf->msg->notice(8) << "Updated image list: " << m_image_list.size() << " images" << std::endl;
    if (m_prof) m_prof->pop();

    return m_image_list;
    }

template <class Shape>
void IntegratorMCMMono<Shape>::updateCellWidth()
    {
    m_nominal_width = getMaxCoreDiameter();

    if (m_patch)
        {
        Scalar max_extent = 0.0;
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            max_extent = std::max(max_extent, m_patch->getAdditiveCutoff(typ));
            }

        m_nominal_width = std::max(m_nominal_width, max_extent+m_patch->getRCut());
        }

    // changing the cell width means that the particle shapes have changed, assume this invalidates the
    // image list and aabb tree
    m_image_list_valid = false;
    m_aabb_tree_invalid = true;
    }

template <class Shape>
void IntegratorMCMMono<Shape>::growAABBList(unsigned int N)
    {
    if (N > m_aabbs_capacity)
        {
        m_aabbs_capacity = N;
        if (m_aabbs != NULL)
            free(m_aabbs);

        int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(detail::AABB));
        if (retval != 0)
            {
            m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
            throw std::runtime_error("Error allocating AABB memory");
            }
        }
    }


/*! Call any time an up to date AABB tree is needed. IntegratorMCMMono internally tracks whether
    the tree needs to be rebuilt or if the current tree can be used.

    buildAABBTree() relies on the member variable m_aabb_tree_invalid to work correctly. Any time particles
    are moved (and not updated with m_aabb_tree->update()) or the particle list changes order, m_aabb_tree_invalid
    needs to be set to true. Then buildAABBTree() will know to rebuild the tree from scratch on the next call. Typically
    this is on the next timestep. But in same cases (i.e. NPT), the tree may need to be rebuilt several times in a
    single n because of box volume moves.

    Subclasses that override update() or other methods must be user to set m_aabb_tree_invalid appropriately, or
    erroneous simulations will result.

    \returns A reference to the tree.
*/
template <class Shape>
const detail::AABBTree& IntegratorMCMMono<Shape>::buildAABBTree()
    {
    if (m_aabb_tree_invalid)
        {
        m_exec_conf->msg->notice(8) << "Building AABB tree: " << m_pdata->getN() << " ptls " << m_pdata->getNGhosts() << " ghosts" << std::endl;
        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "AABB tree build");
        // build the AABB tree
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

            // grow the AABB list to the needed size
            unsigned int n_aabb = m_pdata->getN()+m_pdata->getNGhosts();
            if (n_aabb > 0)
                {
                growAABBList(n_aabb);
                for (unsigned int cur_particle = 0; cur_particle < n_aabb; cur_particle++)
                    {
                    unsigned int i = cur_particle;
                    unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);
                    Shape shape(quat<Scalar>(h_orientation.data[i]), m_params[typ_i]);

                    if (!this->m_patch)
                        m_aabbs[i] = shape.getAABB(vec3<Scalar>(h_postype.data[i]));
                    else
                        {
                        Scalar radius = std::max(0.5*shape.getCircumsphereDiameter(),
                            0.5*this->m_patch->getAdditiveCutoff(typ_i));
                        m_aabbs[i] = detail::AABB(vec3<Scalar>(h_postype.data[i]), radius);
                        }
                    }
                m_aabb_tree.buildTree(m_aabbs, n_aabb);
                }
            }

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
        }

    m_aabb_tree_invalid = false;
    return m_aabb_tree;
    }

/*! Call to reduce the m_d values down to safe levels for the bvh tree + small box limitations. That code path
    will not work if particles can wander more than one image in a time n.

    In MPI simulations, they may not move more than half a local box length.
*/
template <class Shape>
void IntegratorMCMMono<Shape>::limitMoveDistances()
    {
    Scalar3 npd_global = m_pdata->getGlobalBox().getNearestPlaneDistance();
    Scalar min_npd = detail::min(npd_global.x, npd_global.y);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        min_npd = detail::min(min_npd, npd_global.z);
        }

    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        if (m_nselect * h_d.data[typ] > min_npd)
            {
            h_d.data[typ] = min_npd / Scalar(m_nselect);
            m_exec_conf->msg->warning() << "Move distance or nselect too big, reducing move distance to "
                                        << h_d.data[typ] << " for type " << m_pdata->getNameByType(typ) << std::endl;
            m_image_list_valid = false;
            }
        // Sanity check should be performed in code where parameters can be adjusted.
        if (h_d.data[typ] < Scalar(0.0))
            {
            m_exec_conf->msg->warning() << "Move distance has become negative for type " << m_pdata->getNameByType(typ)
                                        << ". This should never happen. Please file a bug report." << std::endl;
            h_d.data[typ] = Scalar(0.0);
            }
        }
    }

/*! Function for finding all overlaps in a system by particle tag. returns an unraveled form of an NxN matrix
 * with true/false indicating the overlap status of the ith and jth particle
 */
template <class Shape>
std::vector<bool> IntegratorMCMMono<Shape>::mapOverlaps()
    {
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_exec_conf->msg->error() << "map_overlaps does not support MPI parallel jobs" << std::endl;
        throw std::runtime_error("map_overlaps does not support MPI parallel jobs");
        }
    #endif

    unsigned int N = m_pdata->getN();

    std::vector<bool> overlap_map(N*N, false);

    m_exec_conf->msg->notice(10) << "MCM overlap mapping" << std::endl;

    unsigned int err_count = 0;

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // Loop over all particles
    for (unsigned int i = 0; i < N; i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        Shape shape_i(quat<Scalar>(orientation_i), m_params[__scalar_as_int(postype_i.w)]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // Check particle against AABB tree for neighbors
        detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                {
                                continue;
                                }

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            Shape shape_j(quat<Scalar>(orientation_j), m_params[__scalar_as_int(postype_j.w)]);

                            if (h_tag.data[i] <= h_tag.data[j]
                                && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                && test_overlap(r_ij, shape_i, shape_j, err_count)
                                && test_overlap(-r_ij, shape_j, shape_i, err_count))
                                {
                                overlap_map[h_tag.data[j]+N*h_tag.data[i]] = true;
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }
                } // end loop over AABB nodes
            } // end loop over images
        } // end loop over particles
    return overlap_map;
    }

/*! Function for returning a python list of all overlaps in a system by particle
  tag. returns an unraveled form of an NxN matrix with true/false indicating
  the overlap status of the ith and jth particle
 */
template <class Shape>
pybind11::list IntegratorMCMMono<Shape>::PyMapOverlaps()
    {
    std::vector<bool> v = IntegratorMCMMono<Shape>::mapOverlaps();
    pybind11::list overlap_map;
    // for( unsigned int i = 0; i < sizeof(v)/sizeof(v[0]); i++ )
    for (auto i: v)
        {
        overlap_map.append(pybind11::cast<bool>(i));
        }
    return overlap_map;
    }

template <class Shape>
void IntegratorMCMMono<Shape>::connectGSDSignal(
                                                    std::shared_ptr<GSDDumpWriter> writer,
                                                    std::string name)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&IntegratorMCMMono<Shape>::slotWriteGSD, this, std::placeholders::_1, name);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template <class Shape>
int IntegratorMCMMono<Shape>::slotWriteGSD( gsd_handle& handle, std::string name ) const
    {
    m_exec_conf->msg->notice(10) << "IntegratorMCMMono writing to GSD File to name: "<< name << std::endl;
    int retval = 0;
    // create schema helpers
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif
    gsd_schema_mcm schema(m_exec_conf, mpi);
    gsd_shape_schema<typename Shape::param_type> schema_shape(m_exec_conf, mpi);

    // access parameters
    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);
    schema.write(handle, "state/mcm/integrate/d", m_pdata->getNTypes(), h_d.data, GSD_TYPE_DOUBLE);
    if(m_hasOrientation)
        {
        schema.write(handle, "state/mcm/integrate/a", m_pdata->getNTypes(), h_a.data, GSD_TYPE_DOUBLE);
        }
    retval |= schema_shape.write(handle, name, m_pdata->getNTypes(), m_params);

    return retval;
    }

template <class Shape>
bool IntegratorMCMMono<Shape>::restoreStateGSD( std::shared_ptr<GSDReader> reader, std::string name)
    {
    bool success = true;
    m_exec_conf->msg->notice(10) << "IntegratorMCMMono from GSD File to name: "<< name << std::endl;
    uint64_t frame = reader->getFrame();
    // create schemas
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif
    gsd_schema_mcm schema(m_exec_conf, mpi);
    gsd_shape_schema<typename Shape::param_type> schema_shape(m_exec_conf, mpi);

    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::readwrite);
    schema.read(reader, frame, "state/mcm/integrate/d", m_pdata->getNTypes(), h_d.data, GSD_TYPE_DOUBLE);
    if(m_hasOrientation)
        {
        schema.read(reader, frame, "state/mcm/integrate/a", m_pdata->getNTypes(), h_a.data, GSD_TYPE_DOUBLE);
        }
    schema_shape.read(reader, frame, name, m_pdata->getNTypes(), m_params);
    return success;
    }

//! Export the IntegratorMCMMono class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorMCMMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorMCMMono(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< IntegratorMCMMono<Shape>, std::shared_ptr< IntegratorMCMMono<Shape> > >(m, name.c_str(), pybind11::base<IntegratorMCM>())
          .def(pybind11::init< std::shared_ptr<SystemDefinition>, unsigned int >())
          .def("setParam", &IntegratorMCMMono<Shape>::setParam)
          .def("setOverlapChecks", &IntegratorMCMMono<Shape>::setOverlapChecks)
          .def("setExternalField", &IntegratorMCMMono<Shape>::setExternalField)
          .def("setPatchEnergy", &IntegratorMCMMono<Shape>::setPatchEnergy)
          .def("mapOverlaps", &IntegratorMCMMono<Shape>::PyMapOverlaps)
          .def("connectGSDSignal", &IntegratorMCMMono<Shape>::connectGSDSignal)
          .def("restoreStateGSD", &IntegratorMCMMono<Shape>::restoreStateGSD)
          ;
    }

template<class Shape>
void IntegratorMCMMono<Shape>::diffuseConductivity(Scalar contactFactor)
    {
    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    const int N=m_pdata->getN();
    const unsigned int ndim = this->m_sysdef->getNDimensions();
    const BoxDim& box = m_pdata->getBox();
    const BoxDim& curBox = m_pdata->getGlobalBox();
    const Scalar3& box_L = curBox.getL(); //save current box dimensions
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);
    ArrayHandle<mcm_counters_t> h_counters(m_count_total, access_location::host, access_mode::readwrite);
    const vec3<Scalar> defaultOrientation2D(0,1,0); //default long axis for 2D spherocylinders
    const vec3<Scalar> defaultOrientation3D(0,0,1); //default long axis for 3D spherocylinders
    const double tiny=1e-7;
    const int runs=20;
    const int steps=1000000;
    // const int max_contacts=20;

    // double contact=0.001*box_L.x;

    // std::cout<<contactFactor<<std::endl;

    double Lx_2=box_L.x/2.0;
    double Ly_2=box_L.y/2.0;
    double Lz_2=box_L.z/2.0;

    const double con1=1;
    const double con2=0;
    const double con3=0;

    double t_arr[steps];
    double r_arr[steps];

    int nbins[3]={100,1000,10000};

    int tmax=steps;

    int dt0=tmax/nbins[0];
    int dt1=tmax/nbins[1];
    int dt2=tmax/nbins[2];

    if (dt0<=0)
        {
        dt0=1;
        }
    if (dt1<=0)
        {
        dt1=1;
        }
    if (dt2<=0)
        {
        dt2=1;
        }

    double ravg0[nbins[0]];
    double ravg1[nbins[1]];
    double ravg2[nbins[2]];

    double ravg0P[nbins[0]];
    double ravg1P[nbins[1]];
    double ravg2P[nbins[2]];

    for (int ii=0;ii<nbins[0];ii++)
        {
        ravg0[ii]=0;
        ravg0P[ii]=0;
        }
    for (int ii=0;ii<nbins[1];ii++)
        {
        ravg1[ii]=0;
        ravg1P[ii]=0;
        }
    for (int ii=0;ii<nbins[2];ii++)
        {
        ravg2[ii]=0;
        ravg2P[ii]=0;
        }

    double sigma=0;

    std::vector<int> neighbors;
    std::vector<std::vector<int> > neighborList(N, neighbors);

    std::vector<vec3<Scalar> > conpoint(0,vec3<Scalar>(0,0,0));
    std::vector<std::vector<vec3<Scalar>>>contactPoints(N,conpoint);

    std::vector<int> starting_points;

    //Initialize displacement variables
    double dx=0;
    double dy=0;
    double dz=0;

    double dx_s=0;
    double dy_s=0;
    double dz_s=0;

    //Same thing, but for contact points
    double dxP=0;
    double dyP=0;
    double dzP=0;

    double dxP_s=0;
    double dyP_s=0;
    double dzP_s=0;

    int ttt=0;
    double r2=0;
    //Contact point r^2
    double r2P=0;

    //initialize number of dt intervals
    int idt0=-1;
    int idt1=-1;
    int idt2=-1;

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<int> gen2(0, N-1); // uniform, unbiased

    // loop through N particles in a shuffled order
    for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
        {
        unsigned int neighbor=0;
        unsigned int i = cur_particle;//m_update_order[cur_particle];
        std::vector<double> k_list;

        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        vec3<Scalar> or_vect_i(0,0,0);

        quat<Scalar> or_i=quat<Scalar>(orientation_i);
        if (ndim==2)
            {
            or_vect_i=rotate(or_i,defaultOrientation2D);
            }
        else if (ndim==3)
            {
            or_vect_i=rotate(or_i,defaultOrientation3D);
            }

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // only move particle if active
            if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                continue;
            }
        #endif

        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
        double radius_i=m_params[typ_i].sweep_radius;

        double length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
        (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
        (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));

        std::vector<std::vector<Scalar>> tmp = getContactDistance(i);

        while (tmp[0].size()==0)
            {
            neighbor=0;
            i = m_update_order[cur_particle];

            // read in the current position and orientation
            postype_i = h_postype.data[i];
            orientation_i = h_orientation.data[i];
            pos_i = vec3<Scalar>(postype_i);

            vec3<Scalar> or_vect_i(0,0,0);

            or_i=quat<Scalar>(orientation_i);
            if (ndim==2)
                {
                or_vect_i=rotate(or_i,defaultOrientation2D);
                }
            else if (ndim==3)
                {
                or_vect_i=rotate(or_i,defaultOrientation3D);
                }

            #ifdef ENABLE_MPI
            if (m_comm)
                {
                // only move particle if active
                if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                    continue;
                }
            #endif

            typ_i = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
            radius_i=m_params[typ_i].sweep_radius;

            length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
            (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
            (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));

            tmp = getContactDistance(i);
            }

        unsigned int neighborIDs[tmp[0].size()];
        Scalar siArr[tmp[0].size()];
        Scalar sjArr[tmp[0].size()];

        vec3<Scalar> images[tmp[0].size()];

        std::vector<Scalar>::iterator result = std::min_element(tmp[1].begin(), tmp[1].end());
        int min_index = std::distance(tmp[1].begin(), result);

        unsigned int min_id = tmp[0][min_index];
        Scalar rmin = tmp[1][min_index];


        for (int q=0;(int) q<tmp[0].size();q++)
            {
            unsigned int ID = (unsigned int) tmp[0][q];
            Scalar si1 = tmp[1][q];
            Scalar sj1 = tmp[2][q];

            double pos_i_image_x = tmp[3][q];
            double pos_i_image_y = tmp[4][q];
            double pos_i_image_z = tmp[5][q];

            unsigned int typ_jj = tmp[6][q];

            vec3<Scalar> pos_i_image_1(pos_i_image_x,pos_i_image_y,pos_i_image_z);

            neighborIDs[q] = ID;
            siArr[q] = si1;
            sjArr[q] = sj1;
            images[q] = pos_i_image_1;
            }

        for (int q=0;q<tmp[0].size();q++)
            {

            vec3<Scalar> pos_i_image_tmp = images[q];
            Scalar si_tmp = siArr[q];
            Scalar sj_tmp = sjArr[q];
            unsigned int j_tmp = neighborIDs[q];

            Scalar4 orientation_j = h_orientation.data[j_tmp];
            quat<Scalar> or_j=quat<Scalar>(orientation_j);
            Scalar4 postype_j = h_postype.data[j_tmp];
            vec3<Scalar> pos_j = vec3<Scalar>(postype_j);
            or_j=quat<Scalar>(orientation_j);

            unsigned int typ_j = __scalar_as_int(postype_j.w);
            double radius_j=m_params[typ_j].sweep_radius;

            double length_j=sqrt((m_params[typ_j].x[0]-m_params[typ_j].x[1])*(m_params[typ_j].x[0]-m_params[typ_j].x[1])+
                            (m_params[typ_j].y[0]-m_params[typ_j].y[1])*(m_params[typ_j].y[0]-m_params[typ_j].y[1])+
                            (m_params[typ_j].z[0]-m_params[typ_j].z[1])*(m_params[typ_j].z[0]-m_params[typ_j].z[1]));

            vec3<Scalar> or_vect_i;
            vec3<Scalar> or_vect_j;
            if (ndim==3)
                {
                or_vect_i=rotate(or_i,defaultOrientation3D);
                or_vect_j=rotate(or_j,defaultOrientation3D);
                }

            else if (ndim==2)
                {
                or_vect_i=rotate(or_i,defaultOrientation2D);
                or_vect_j=rotate(or_j,defaultOrientation2D);
                }

            vec3<Scalar> r_ij_tmp = vec3<Scalar>(postype_j) - pos_i_image_tmp;

            vec3<Scalar> k_vect=(pos_i_image_tmp+si_tmp*or_vect_i)-(pos_j+sj_tmp*or_vect_j);

            // vec3<Scalar> k_vect=(pos_i_image+si*or_vect_i)-(pos_j+sj*or_vect_j);
            // vec3<Scalar> k_vect=(pos_i+si*or_vect_i)-(pos_j+sj*or_vect_j);

            double mag_k=sqrt(dot(k_vect,k_vect));

            vec3<Scalar> contactPoint=(pos_i_image_tmp+si_tmp*or_vect_i)-((radius_i/mag_k)*k_vect);

            double delta=(radius_i+radius_j)-mag_k;

            double contact=radius_i*contactFactor;

            if (delta>-contact && i!=j_tmp)
                {
                neighborList[i].push_back(j_tmp);
                contactPoints[i].push_back(contactPoint);
                neighbor++;
                }
            }
        }

    //Make a list of valid starting IDs
    //Must be conductive particles with at least 1 conductive neighbor
    std::vector<int> neighborNums;
    for (int qq=0;qq<N;qq++)
        {
        bool validStart=false;
        int n_neighbors=neighborList[qq].size();
        if (n_neighbors!=0)
            {
            for (int zz=0;zz<n_neighbors;zz++)
                {
                unsigned int jj=neighborList[qq][zz];
                Scalar4 postype_jj= h_postype.data[jj];
                unsigned int typ_jj = __scalar_as_int(postype_jj.w);
                if (typ_jj==0)
                    {
                    validStart=true;
                    }
                }
            if (validStart==true)
                {
                neighborNums.push_back(qq);
                }
            }
        }

    int possibleStarts=neighborNums.size();

    unsigned int i_prev;
    int index_prev;

    // loop through runs particles in a shuffled order
    for (unsigned int cur_particle2 = 0; cur_particle2 < runs; cur_particle2++)
        {
        // std::cout<<cur_particle2<<std::endl;
        //reset displacements at the start of each walk
        dx=0;
        dy=0;
        dz=0;

        dxP=0;
        dyP=0;
        dzP=0;

        ttt=0;
        r2=0;
        r2P=0;

        int ii;

        // std::cout<<"before break"<<std::endl;

        //Initialize starting particle
        if (possibleStarts==0)
            {
            std::cout<<"ERROR: NO VALID STARTING POINTS"<<std::endl;
            break;
            }
        else if (possibleStarts==1)
            {
            ii=0;
            }
        else
            {
            std::uniform_int_distribution<int> gen3(0, possibleStarts-1);
            ii = (unsigned int) gen3(rng);
            }
        unsigned int i = neighborNums[ii];

        Scalar4 postype_i = h_postype.data[i];
        unsigned int typ_i = __scalar_as_int(postype_i.w);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);


        //ensure we start on a conductive site
        // while (typ_i!=0)
        //     {
        //     if (possibleStarts==0)
        //         {
        //         std::cout<<"ERROR: NO VALID STARTING POINTS"<<std::endl;
        //         break;
        //         }
        //     else if (possibleStarts==1)
        //         {
        //         ii=0;
        //         }
        //     else
        //         {
        //         std::uniform_int_distribution<int> gen3(0, possibleStarts-1);
        //         ii =  gen3(rng);
        //         }
        //     i = neighborNums[ii];
        //     Scalar4 postype_i = h_postype.data[i];
        //     unsigned int typ_i = __scalar_as_int(postype_i.w);
        //     vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
        //     }

        // starting_points.push_back(i);

        i_prev=i;

        index_prev=0;

        idt0=-1;
        idt1=-1;
        idt2=-1;

        // std::cout<<"after break"<<std::endl;

        //Once we choose starting point, random walk through for steps
        for (unsigned int n=0;n<steps;n++)
            {
            // if (n%100000==0)
            //     {
            //     std::cout<<n<<std::endl;
            //     }
            // dx_s=0;
            // dy_s=0;
            // dz_s=0;

            // dxP_s=0;
            // dyP_s=0;
            // dzP_s=0;

            // read in the current position and orientation
            postype_i = h_postype.data[i];
            pos_i = vec3<Scalar>(postype_i);
            typ_i = __scalar_as_int(postype_i.w);

            //pick a random neighbor
            int n_neighbors=neighborList[i].size();
            int index;
            if (n_neighbors==1)
                {
                index=0;
                }
            else
                {
                std::uniform_int_distribution<int> gen(0, n_neighbors-1); // uniform, unbiased
                index = gen(rng);
                }
            unsigned int j=neighborList[i][index];

            //Read in its properties
            Scalar4 postype_j= h_postype.data[j];
            unsigned int typ_j = __scalar_as_int(postype_j.w);
            vec3<Scalar> pos_j = vec3<Scalar>(postype_j);

            //Increment time every step
            ttt+=1;

            if (typ_j!=0)
                {
                dx_s=0.0;
                dy_s=0.0;
                dz_s=0.0;

                dxP_s=0.0;
                dyP_s=0.0;
                dzP_s=0.0;
                }

            else
                {
                //Default to in-box measurement
                dx_s=pos_j.x-pos_i.x;
                dy_s=pos_j.y-pos_i.y;
                dz_s=pos_j.z-pos_i.z;

                //No previous particle in the first iteration
                if (n==0)
                    {
                    dxP_s=0;
                    dyP_s=0;
                    dzP_s=0;
                    }

                else
                    {
                    dxP_s=contactPoints[i][index].x-contactPoints[i_prev][index_prev].x;
                    dyP_s=contactPoints[i][index].y-contactPoints[i_prev][index_prev].y;
                    dzP_s=contactPoints[i][index].z-contactPoints[i_prev][index_prev].z;
                    }
                }

            //Subtract box size is particles cross periodic boundaries
            if (dx_s>=Lx_2)
                    {
                    dx_s=dx_s-box_L.x;
                    }
            else if(dx_s<=-Lx_2)
                    {
                    dx_s=dx_s+box_L.x;
                    }
            if (dy_s>=Ly_2)
                    {
                    dy_s=dy_s-box_L.y;
                    }
            else if(dy_s<=-Ly_2)
                    {
                    dy_s=dy_s+box_L.y;
                    }
            if (dz_s>=Lz_2)
                    {
                    dz_s=dz_s-box_L.z;
                    }
            else if(dz_s<=-Lz_2)
                    {
                    dz_s=dz_s+box_L.z;
                    }

            //Contact point equivalent
            if (dxP_s>=Lx_2)
                    {
                    dxP_s=dxP_s-box_L.x;
                    }
            else if(dxP_s<=-Lx_2)
                    {
                    dxP_s=dxP_s+box_L.x;
                    }
            if (dyP_s>=Ly_2)
                    {
                    dyP_s=dyP_s-box_L.y;
                    }
            else if(dyP_s<=-Ly_2)
                    {
                    dyP_s=dyP_s+box_L.y;
                    }
            if (dzP_s>=Lz_2)
                    {
                    dzP_s=dzP_s-box_L.z;
                    }
            else if(dzP_s<=-Lz_2)
                    {
                    dzP_s=dzP_s+box_L.z;
                    }

            //Update displacements
            dx=dx+dx_s;
            dy=dy+dy_s;
            dz=dz+dz_s;

            dxP=dxP+dxP_s;
            dyP=dyP+dyP_s;
            dzP=dzP+dzP_s;

            //on a sucessful move, change to new particle
            i_prev=i;
            index_prev=index;
            i=j;

            //Calculate r^2 when needed
            if (ttt%dt0==0 || ttt%dt1==0 || ttt%dt2==0)
                {
                r2=dx*dx+dy*dy+dz*dz;
                r2P=dxP*dxP+dyP*dyP+dzP*dzP;
                }
            //Average every dt steps
            if (ttt%dt0==0)
                {
                idt0+=1; //update number of dt intervals
                // std::ofstream outfile_avg0;
                // outfile_avg0.open("avg0.txt", std::ios_base::app);
                ravg0[idt0]+=r2;
                ravg0P[idt0]+=r2P;
                // outfile_avg0<<ttt<<' '<<r2<<std::endl;
                // outfile_avg0.close();
                }
            if (ttt%dt1==0)
                {
                idt1+=1; //update number of dt intervals
                // std::ofstream outfile_avg1;
                // outfile_avg1.open("avg1.txt", std::ios_base::app);
                ravg1[idt1]+=r2;
                ravg1P[idt1]+=r2P;
                // outfile_avg1<<ttt<<' '<<r2<<std::endl;
                // outfile_avg1.close();
                }
            if (ttt%dt2==0)
                {
                idt2+=1; //update number of dt intervals
                // std::ofstream outfile_avg2;
                // outfile_avg2.open("avg2.txt", std::ios_base::app);
                ravg2[idt2]+=r2;
                ravg2P[idt2]+=r2P;
                // outfile_avg2<<ttt<<' '<<r2<<std::endl;
                // outfile_avg2.close();
                }
            }//end loop over steps
        // std::cout<<"after steps"<<std::endl;
        // std::cout<<idt0+1<<' '<<idt1+1<<' '<<idt2+1<<std::endl;
        } // end loop over runs
    // std::cout<<"after runs"<<std::endl;

    //Average every dt steps and output files
    double tt=0;
    std::ofstream outfile_dt0;
    outfile_dt0.open("diffuse_data_dt"+std::to_string(nbins[0])+".txt", std::ios_base::app);
    for (int k=0; k<=idt0; k++)
        {
        ravg0[k]=ravg0[k]/(runs);
        ravg0P[k]=ravg0P[k]/(runs);
        tt=(k+1)*dt0;
        outfile_dt0<<tt<<" "<<ravg0[k]<<" "<<ravg0P[k]<<" "<<contactFactor<<std::endl;
        }
    outfile_dt0.close();

    std::ofstream outfile_dt1;
    outfile_dt1.open("diffuse_data_dt"+std::to_string(nbins[1])+".txt", std::ios_base::app);
    for (int k=0; k<=idt1; k++)
        {
        ravg1[k]=ravg1[k]/(runs);
        ravg1P[k]=ravg1P[k]/(runs);
        tt=(k+1)*dt1;
        outfile_dt1<<tt<<" "<<ravg1[k]<<" "<<ravg1P[k]<<" "<<contactFactor<<std::endl;
        }
    outfile_dt1.close();

    std::ofstream outfile_dt2;
    outfile_dt2.open("diffuse_data_dt"+std::to_string(nbins[2])+".txt", std::ios_base::app);
    for (int k=0; k<=idt2; k++)
        {
        ravg2[k]=ravg2[k]/(runs);
        ravg2P[k]=ravg2P[k]/(runs);
        tt=(k+1)*dt2;
        outfile_dt2<<tt<<" "<<ravg2[k]<<" "<<ravg2P[k]<<" "<<contactFactor<<std::endl;
        }
    outfile_dt2.close();
    }

template<class Shape>
void IntegratorMCMMono<Shape>::writePairs(Scalar contactFactor)
    {
    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    const int nTypes=m_pdata->getNTypes();
    const int N=m_pdata->getN();
    const int maxCoordN=100; //highest coordination number
    const BoxDim box = m_pdata->getBox();
    const Scalar3 box_L = box.getL();
    const unsigned int ndim = this->m_sysdef->getNDimensions();
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);
    ArrayHandle<mcm_counters_t> h_counters(m_count_total, access_location::host, access_mode::readwrite);
    const vec3<Scalar> defaultOrientation2D(0,1,0); //default long axis for 2D spherocylinders
    const vec3<Scalar> defaultOrientation3D(0,0,1); //default long axis for 3D spherocylinders
    const double tiny=1e-7;
    const double tol=1;
    // int single_contacts=0;
    // double contact=0.001*box_L.x;
    int perc_direction=-1;

    unsigned int* pair_list = new unsigned int[nTypes*N*maxCoordN*2];

    for (int i=0;i<nTypes;i++)
        {
        for (int j=0;j<N*maxCoordN;j++)
            {
            for (int k=0;k<2;k++)
                {
                pair_list[i*N*maxCoordN*2+j*2+k]=(unsigned int) 0;
                }
            }
        }
    // loop through N particles in a shuffled order
    unsigned int n_pairs=0;

    for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
        {
        unsigned int i = cur_particle;//m_update_order[cur_particle];
        // single_contacts=0;

        std::vector<Scalar> neighbor_dists;

        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        vec3<Scalar> or_vect_i(0,0,0);

        quat<Scalar> or_i=quat<Scalar>(orientation_i);
        if (ndim==2)
            {
            or_vect_i=rotate(or_i,defaultOrientation2D);
            }
        else if (ndim==3)
            {
            or_vect_i=rotate(or_i,defaultOrientation3D);
            }

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // only move particle if active
            if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                continue;
            }
        #endif

        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);

        double radius_i=m_params[typ_i].sweep_radius;

        double length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
        (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
        (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));

        std::vector<std::vector<Scalar>> tmp = getContactDistance(i);

        unsigned int neighborIDs[tmp[0].size()];
        Scalar siArr[tmp[0].size()];
        Scalar sjArr[tmp[0].size()];

        vec3<Scalar> images[tmp[0].size()];

        std::vector<Scalar>::iterator result = std::min_element(tmp[1].begin(), tmp[1].end());
        int min_index = std::distance(tmp[1].begin(), result);

        unsigned int min_id = tmp[0][min_index];
        Scalar rmin = tmp[1][min_index];
        unsigned int typ_j;


        for (int q=0;q<tmp[0].size();q++)
            {
            unsigned int ID = (unsigned int) tmp[0][q];
            Scalar si1 = tmp[1][q];
            Scalar sj1 = tmp[2][q];

            double pos_i_image_x = tmp[3][q];
            double pos_i_image_y = tmp[4][q];
            double pos_i_image_z = tmp[5][q];

            vec3<Scalar> pos_i_image_1(pos_i_image_x,pos_i_image_y,pos_i_image_z);

            neighborIDs[q] = ID;
            siArr[q] = si1;
            sjArr[q] = sj1;
            images[q] = pos_i_image_1;
            }

        for (int q=0;q<tmp[0].size();q++)
            {

            vec3<Scalar> pos_i_image_tmp = images[q];
            Scalar si_tmp = siArr[q];
            Scalar sj_tmp = sjArr[q];
            unsigned int j_tmp = neighborIDs[q];

            Scalar4 orientation_j = h_orientation.data[j_tmp];
            quat<Scalar> or_j=quat<Scalar>(orientation_j);
            Scalar4 postype_j = h_postype.data[j_tmp];
            vec3<Scalar> pos_j = vec3<Scalar>(postype_j);
            or_j=quat<Scalar>(orientation_j);

            typ_j = __scalar_as_int(postype_j.w);
            double radius_j=m_params[typ_j].sweep_radius;

            double length_j=sqrt((m_params[typ_j].x[0]-m_params[typ_j].x[1])*(m_params[typ_j].x[0]-m_params[typ_j].x[1])+
                            (m_params[typ_j].y[0]-m_params[typ_j].y[1])*(m_params[typ_j].y[0]-m_params[typ_j].y[1])+
                            (m_params[typ_j].z[0]-m_params[typ_j].z[1])*(m_params[typ_j].z[0]-m_params[typ_j].z[1]));

            vec3<Scalar> or_vect_i;
            vec3<Scalar> or_vect_j;
            if (ndim==3)
                {
                or_vect_i=rotate(or_i,defaultOrientation3D);
                or_vect_j=rotate(or_j,defaultOrientation3D);
                }

            else if (ndim==2)
                {
                or_vect_i=rotate(or_i,defaultOrientation2D);
                or_vect_j=rotate(or_j,defaultOrientation2D);
                }

            vec3<Scalar> r_ij_tmp = vec3<Scalar>(postype_j) - pos_i_image_tmp;

            vec3<Scalar> k_vect=(pos_i_image_tmp+si_tmp*or_vect_i)-(pos_j+sj_tmp*or_vect_j);

            double mag_k=sqrt(dot(k_vect,k_vect));
            double delta=(radius_i+radius_j)-mag_k;

            neighbor_dists.push_back(delta);

            // if (delta>-length_i/2.0 && i!=j_tmp)
            //     {
            //     single_contacts++;
            //     }

            double contact=radius_i*contactFactor;

            if (delta>-contact && typ_i==typ_j && i!=j_tmp) //particles are overlapping
                {
                pair_list[typ_i*N*maxCoordN*2+n_pairs*2+0]=i;
                pair_list[typ_j*N*maxCoordN*2+n_pairs*2+1]=j_tmp;
                n_pairs++;
                }
            }
            std::ofstream outfile1;
            outfile1.open("contact_stats.txt", std::ios_base::app);
            // outfile1<<single_contacts<<std::endl;
            // outfile1<<"---"<<std::endl;
            for (int z=0;z<neighbor_dists.size();z++)
                {
                if (neighbor_dists[z]>-radius_i)
                    {
                    outfile1<<neighbor_dists[z]<<" "<<typ_i<<" "<<typ_j<<std::endl;
                    }
                }
            outfile1<<std::endl;
            outfile1.close();
            neighbor_dists.clear();
            // single_contacts=0;
        } // end loop over all particles

    for (int type=0;type<nTypes;type++) //find clusters of each type
        {
        double maxdist_x=-999;
        double maxdist_y=-999;
        double maxdist_z=-999;
        double radius=m_params[(unsigned int) type].sweep_radius;
        int percolating = 0;
        std::vector<int> clusters;
        std::vector<int> percolating_clusters;
        for (int i=0;i<N*maxCoordN;i++)
            {
            if (pair_list[type*N*maxCoordN*2+i*2+0] == 0 && pair_list[type*N*maxCoordN*2+i*2+1]==0)
                {
                continue;
                }
            else
                {
                clusters.push_back(i); //start with each pair as its own cluster
                }
            }
        int cluster_number=clusters.size();

        if (cluster_number==0) //catch situations where a type has no pairs
            {
            std::ofstream outfile;

            outfile.open(m_pdata->getNameByType(type)+"_perc.txt", std::ios_base::app);
            outfile << percolating<<std::endl;
            outfile.close();
            }
        else
            {
            bool done=false;
            int tries=0;
            // iteratively group pairs that share elements into clusters
            while (done==false || tries<3)
                {
                for (int i=0;i<(int) clusters.size();i++)
                    {
                    for (int j=i+1;j<(int) clusters.size();j++)
                        {
                        if (clusters[i]!=clusters[j])
                            {
                            //combine clusters that share common elements
                            if (pair_list[type*N*maxCoordN*2+i*2+0]==pair_list[type*N*maxCoordN*2+j*2+0] ||
                                pair_list[type*N*maxCoordN*2+i*2+1]==pair_list[type*N*maxCoordN*2+j*2+1] ||
                                pair_list[type*N*maxCoordN*2+i*2+0]==pair_list[type*N*maxCoordN*2+j*2+1] ||
                                pair_list[type*N*maxCoordN*2+i*2+1]==pair_list[type*N*maxCoordN*2+j*2+0])

                                {
                                int name=std::min(clusters[i],clusters[j]);
                                clusters[i]=name;
                                clusters[j]=name;
                                }
                            }
                        }
                    }
                std::vector<int> names;
                for (int k=0;k<(int) clusters.size();k++)
                    {
                    int name=clusters[k];
                    //check each cluster name to see if its in the list of names
                    if (std::find(names.begin(), names.end(), name) == names.end())
                        {
                        names.push_back(name);
                        }
                    //if number of clusters not changing, probably done
                    if (cluster_number==(int) names.size())
                        {
                        done=true;
                        tries+=1; //error catching
                        }
                    //if number of clusters decreases, not done
                    else if (cluster_number>(int) names.size())
                        {
                        cluster_number=(int) names.size();
                        done=false;
                        tries=0;
                        }
                    }
                }
            //construct final list of cluster names
            std::vector<int> old_names;
            for (int k=0;k<(int) clusters.size();k++)
                {
                int name=clusters[k];
                //check each cluster name to see if its in the list of names
                if (std::find(old_names.begin(), old_names.end(), name) == old_names.end())
                    {
                    old_names.push_back(name);
                    }
                }

            //rename clusters to be sequential numbers
            for (int j=0;j<(int) clusters.size();j++)
                {
                for (int i=0;i<(int) old_names.size();i++)
                    {
                    if (clusters[j]==old_names[i])
                        {
                        clusters[j]=i;
                        }
                    }
                }

            //reconstruct final list of cluster names, now sequential
            std::vector<int> names;
            for (int k=0;k<(int) clusters.size();k++)
                {
                int name=clusters[k];
                //check each cluster name to see if its in the list of names
                if (std::find(names.begin(), names.end(), name) == names.end())
                    {
                    names.push_back(name);
                    }
                }

            //find maximum extent of each cluster
            int total_clusters=names.size();
            double xdist[total_clusters];
            double ydist[total_clusters];
            double zdist[total_clusters];
            for (int k=0;k<(int) names.size();k++)
                {
                double xmin=999;
                double xmax=-999;
                double ymin=999;
                double ymax=-999;
                double zmin=999;
                double zmax=-999;
                for (int l=0;l<(int) clusters.size();l++)
                    {
                    if (clusters[l]==names[k])
                        {
                        unsigned int i=pair_list[type*N*maxCoordN*2+l*2+0];
                        unsigned int j=pair_list[type*N*maxCoordN*2+l*2+1];
                        Scalar4 postype_i = h_postype.data[i];
                        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
                        unsigned int typ_i=postype_i.w;
                        Scalar4 orientation_i = h_orientation.data[i];
                        quat<Scalar> or_i=quat<Scalar>(orientation_i);
                        Scalar4 postype_j = h_postype.data[j];
                        vec3<Scalar> pos_j = vec3<Scalar>(postype_j);
                        unsigned int typ_j=postype_j.w;
                        Scalar4 orientation_j = h_orientation.data[j];
                        quat<Scalar> or_j=quat<Scalar>(orientation_j);
                        double radius_i=m_params[typ_i].sweep_radius;
                        double radius_j=m_params[typ_j].sweep_radius;


                        vec3<Scalar> or_vect_i;
                        vec3<Scalar> or_vect_j;
                        if (ndim==3)
                            {
                            or_vect_i=rotate(or_i,defaultOrientation3D);
                            or_vect_j=rotate(or_j,defaultOrientation3D);
                            }
                        else if (ndim==2)
                            {
                            or_vect_i=rotate(or_i,defaultOrientation2D);
                            or_vect_j=rotate(or_j,defaultOrientation2D);
                            }

                        double length_i=sqrt((m_params[typ_i].x[0]-m_params[typ_i].x[1])*(m_params[typ_i].x[0]-m_params[typ_i].x[1])+
                        (m_params[typ_i].y[0]-m_params[typ_i].y[1])*(m_params[typ_i].y[0]-m_params[typ_i].y[1])+
                        (m_params[typ_i].z[0]-m_params[typ_i].z[1])*(m_params[typ_i].z[0]-m_params[typ_i].z[1]));

                        double length_j=sqrt((m_params[typ_j].x[0]-m_params[typ_j].x[1])*(m_params[typ_j].x[0]-m_params[typ_j].x[1])+
                        (m_params[typ_j].y[0]-m_params[typ_j].y[1])*(m_params[typ_j].y[0]-m_params[typ_j].y[1])+
                        (m_params[typ_j].z[0]-m_params[typ_j].z[1])*(m_params[typ_j].z[0]-m_params[typ_j].z[1]));

                        vec3<Scalar> top_i=pos_i+(0.5*length_i+radius_i)*or_vect_i;
                        vec3<Scalar> bottom_i=pos_i-(0.5*length_i+radius_i)*or_vect_i;

                        vec3<Scalar> top_j=pos_j+(0.5*length_j+radius_j)*or_vect_j;
                        vec3<Scalar> bottom_j=pos_j-(0.5*length_j+radius_j)*or_vect_j;

                        double xmax_i=std::max(top_i.x,bottom_i.x);
                        double xmin_i=std::min(top_i.x,bottom_i.x);
                        double ymax_i=std::max(top_i.y,bottom_i.y);
                        double ymin_i=std::min(top_i.y,bottom_i.y);
                        double zmax_i=std::max(top_i.z,bottom_i.z);
                        double zmin_i=std::min(top_i.z,bottom_i.z);

                        double xmax_j=std::max(top_j.x,bottom_j.x);
                        double xmin_j=std::min(top_j.x,bottom_j.x);
                        double ymax_j=std::max(top_j.y,bottom_j.y);
                        double ymin_j=std::min(top_j.y,bottom_j.y);
                        double zmax_j=std::max(top_j.z,bottom_j.z);
                        double zmin_j=std::min(top_j.z,bottom_j.z);

                        double xmax_tmp=std::max(xmax_i,xmax_j);
                        double xmin_tmp=std::min(xmin_i,xmin_j);
                        double ymax_tmp=std::max(ymax_i,ymax_j);
                        double ymin_tmp=std::min(ymin_i,ymin_j);
                        double zmax_tmp=std::max(zmax_i,zmax_j);
                        double zmin_tmp=std::min(zmin_i,zmin_j);

                        xmax=std::max(xmax,xmax_tmp);
                        xmin=std::min(xmin,xmin_tmp);
                        ymax=std::max(ymax,ymax_tmp);
                        ymin=std::min(ymin,ymin_tmp);
                        zmax=std::max(zmax,zmax_tmp);
                        zmin=std::min(zmin,zmin_tmp);
                        }
                    xdist[k]=xmax-xmin;
                    ydist[k]=ymax-ymin;
                    zdist[k]=zmax-zmin;
                    }

                for (int r=0;r<total_clusters;r++)
                    {
                    if (xdist[r]>maxdist_x)
                        {
                        maxdist_x=xdist[r];
                        }
                    if (ydist[r]>maxdist_y)
                        {
                        maxdist_y=ydist[r];
                        }
                    if (zdist[r]>maxdist_z)
                        {
                        maxdist_z=zdist[r];
                        }
                    }
                if (maxdist_x+tol*2*radius>box_L.x)
                    {
                    percolating=1;
                    percolating_clusters.push_back(names[k]);
                    perc_direction=0;
                    }
                if (maxdist_y+tol*2*radius>box_L.y)
                    {
                    percolating=1;
                    percolating_clusters.push_back(names[k]);
                    perc_direction=1;
                    }
                if (ndim==3)
                    {
                    if (maxdist_z+tol*2*radius>box_L.z)
                        {
                        percolating=1;
                        percolating_clusters.push_back(names[k]);
                        perc_direction=2;
                        }
                    }
                }
            // double correlation_length=calculateCorrelationLength(clusters, percolating_clusters);

            std::ofstream outfile;

            outfile.open(m_pdata->getNameByType(type)+"_perc.txt", std::ios_base::app);
            // outfile << percolating<<std::endl;
            outfile<<percolating<<" "<<box_L.x<<" "<<avg_contacts<<" "<<perc_direction<<" "<<contactFactor<<std::endl;
            outfile.close();
            }
        }
    delete[] pair_list;
    }

} // end namespace mcm

#endif // _INTEGRATOR_MCM_MONO_H_
