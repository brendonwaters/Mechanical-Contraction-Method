// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _COMPUTE_FREE_VOLUME_CUH_
#define _COMPUTE_FREE_VOLUME_CUH_

#include "MCMCounters.h"
#include "MCMPrecisionSetup.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/Saru.h"

#include <curand_kernel.h>

#ifdef NVCC
#include "Moves.h"
#include "hoomd/TextureTools.h"
#endif

namespace mcm
{

namespace detail
{

/*! \file IntegratorMCMMonoImplicit.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_mcm_free_volume
/*! \ingroup mcm_data_structs */
struct mcm_free_volume_args_t
    {
    //! Construct a pair_args_t
    mcm_free_volume_args_t(
                unsigned int _n_sample,
                unsigned int _type,
                Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                const unsigned int *_d_cell_idx,
                const unsigned int *_d_cell_size,
                const Index3D& _ci,
                const Index2D& _cli,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                const uint3& _cell_dim,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                unsigned int _select,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const unsigned int _max_n,
                unsigned int *_d_n_overlap_all,
                const Scalar3 _ghost_width,
                const unsigned int *_d_check_overlaps,
                Index2D _overlap_idx,
                cudaStream_t _stream,
                const cudaDeviceProp& _devprop
                )
                : n_sample(_n_sample),
                  type(_type),
                  d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_cell_idx(_d_cell_idx),
                  d_cell_size(_d_cell_size),
                  ci(_ci),
                  cli(_cli),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  cell_dim(_cell_dim),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  select(_select),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  max_n(_max_n),
                  d_n_overlap_all(_d_n_overlap_all),
                  ghost_width(_ghost_width),
                  d_check_overlaps(_d_check_overlaps),
                  overlap_idx(_overlap_idx),
                  stream(_stream),
                  devprop(_devprop)
        {
        };

    unsigned int n_sample;            //!< Number of depletants particles to generate
    unsigned int type;                //!< Type of depletant particle
    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    const unsigned int *d_cell_idx;   //!< Index data for each cell
    const unsigned int *d_cell_size;  //!< Number of particles in each cell
    const Index3D& ci;                //!< Cell indexer
    const Index2D& cli;               //!< Indexer for d_cell_idx
    const unsigned int *d_excell_idx; //!< Expanded cell neighbors
    const unsigned int *d_excell_size; //!< Size of expanded cell list per cell
    const Index2D excli;              //!< Expanded cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    unsigned int select;              //!< RNG select value
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int block_size;          //!< Block size to execute
    unsigned int stride;              //!< Number of threads per overlap check
    unsigned int group_size;          //!< Size of the group to execute
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    unsigned int *d_n_overlap_all;    //!< Total number of depletants in overlap volume
    const Scalar3 ghost_width;       //!< Width of ghost layer
    const unsigned int *d_check_overlaps;   //!< Interaction matrix
    Index2D overlap_idx;              //!< Interaction matrix indexer
    cudaStream_t stream;               //!< Stream for kernel execution
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    };

template< class Shape >
cudaError_t gpu_mcm_free_volume(const mcm_free_volume_args_t &args, const typename Shape::param_type *d_params);

#ifdef NVCC
//! Texture for reading postype
scalar4_tex_t free_volume_postype_tex;
//! Texture for reading orientation
scalar4_tex_t free_volume_orientation_tex;

//! Compute the cell that a particle sits in
__device__ inline unsigned int compute_cell_idx(const Scalar3 p,
                                               const BoxDim& box,
                                               const Scalar3& ghost_width,
                                               const uint3& cell_dim,
                                               const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    return ci(ib,jb,kb);
    }


//! Kernel to estimate the colloid overlap volume and the depletant free volume
/*! \param n_sample Number of probe depletant particles to generate
    \param type Type of depletant particle
    \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param d_cell_size The size of each cell
    \param ci Cell indexer
    \param cli Cell list indexer
    \param d_cell_adj List of adjacent cells
    \param cadji Cell adjacency indexer
    \param cell_dim Dimensions of the cell list
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param a Size of rotation move (per type)
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param d_n_overlap_all Total overlap counter (output value)
    \param ghost_width Width of ghost layer
    \param d_params Per-type shape parameters
    \param d_overlaps Per-type pair interaction matrix
*/
template< class Shape >
__global__ void gpu_mcm_free_volume_kernel(unsigned int n_sample,
                                     unsigned int type,
                                     Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     const unsigned int *d_cell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int select,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     unsigned int *d_n_overlap_all,
                                     Scalar3 ghost_width,
                                     const unsigned int *d_check_overlaps,
                                     Index2D overlap_idx,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_extra_bytes)
    {
    unsigned int group = threadIdx.z;
    unsigned int offset = threadIdx.y;
    unsigned int group_size = blockDim.y;
    bool master = (offset == 0 && threadIdx.x == 0);
    unsigned int n_groups = blockDim.z;

    __shared__ unsigned int s_n_overlap;

    // determine sample idx
    unsigned int i;
    if (gridDim.y > 1)
        {
        // if gridDim.y > 1, then the fermi workaround is in place, index blocks on a 2D grid
        i = (blockIdx.x + blockIdx.y * 65535) * n_groups + group;
        }
    else
        {
        i = blockIdx.x * n_groups + group;
        }


    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    unsigned int *s_check_overlaps = (unsigned int *) (s_params + num_types);
    unsigned int ntyppairs = overlap_idx.getNumElements();
    unsigned int *s_overlap = (unsigned int *)(&s_check_overlaps[ntyppairs]);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_overlap + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    if (master)
        {
        s_overlap[group] = 0;
        }

    if (master && group == 0)
        {
        s_n_overlap = 0;
        }

    __syncthreads();

    bool active = true;

    if (i >= n_sample)
        {
        active = false;
        }

    // one RNG per particle
    hoomd::detail::Saru rng(i, seed+select, timestep);

    unsigned int my_cell;

    // test depletant position
    vec3<Scalar> pos_i;
    quat<Scalar> orientation_i;
    Shape shape_i(orientation_i, s_params[type]);

    if (active)
        {
        // select a random particle coordinate in the box
        Scalar xrand = rng.template s<Scalar>();
        Scalar yrand = rng.template s<Scalar>();
        Scalar zrand = rng.template s<Scalar>();

        Scalar3 f = make_scalar3(xrand, yrand, zrand);
        pos_i = vec3<Scalar>(box.makeCoordinates(f));

        if (shape_i.hasOrientation())
            {
            shape_i.orientation = generateRandomOrientation(rng);
            }

        // find cell the particle is in
        Scalar3 p = vec_to_scalar3(pos_i);
        my_cell = compute_cell_idx(p, box, ghost_width, cell_dim, ci);
        }

    if (active)
        {
        // loop over neighboring cells and check for overlaps
        unsigned int excell_size = d_excell_size[my_cell];

        for (unsigned int k = 0; k < excell_size; k += group_size)
            {
            unsigned int local_k = k + offset;
            if (local_k < excell_size)
                {
                // read in position, and orientation of neighboring particle
                #if ( __CUDA_ARCH__ > 300)
                unsigned int j = __ldg(&d_excell_idx[excli(local_k, my_cell)]);
                #else
                unsigned int j = d_excell_idx[excli(local_k, my_cell)];
                #endif

                Scalar4 postype_j = texFetchScalar4(d_postype, free_volume_postype_tex, j);
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[typ_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, free_volume_orientation_tex, j));

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i;
                r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                // check for overlaps
                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                if (rsq*OverlapReal(4.0) <= DaDb * DaDb)
                    {
                    // circumsphere overlap
                    unsigned int err_count;
                    if (s_check_overlaps[overlap_idx(typ_j, type)] && test_overlap(r_ij, shape_i, shape_j, err_count))
                        {
                        atomicAdd(&s_overlap[group],1);
                        break;
                        }
                    }
                }
            }
        }

    __syncthreads();

    unsigned int overlap = s_overlap[group];

    if (master)
        {
        // this thread counts towards the total overlap volume
        if (overlap)
            {
            atomicAdd(&s_n_overlap, 1);
            }
        }

    __syncthreads();

    if (master && group == 0 && s_n_overlap)
        {
        // final tally into global mem
        atomicAdd(d_n_overlap_all, s_n_overlap);
        }
    }

//! Kernel driver for gpu_mcm_free_volume_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for parallel update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup mcm_kernels
*/
template< class Shape >
cudaError_t gpu_mcm_free_volume(const mcm_free_volume_args_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_cell_size);
    assert(args.group_size >= 1);
    assert(args.group_size <= 32);  // note, really should be warp size of the device
    assert(args.block_size%(args.stride*args.group_size)==0);


    // bind the textures
    free_volume_postype_tex.normalized = false;
    free_volume_postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, free_volume_postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    free_volume_orientation_tex.normalized = false;
    free_volume_orientation_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, free_volume_orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    // reset counters
    cudaMemsetAsync(args.d_n_overlap_all,0, sizeof(unsigned int), args.stream);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static int sm = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_mcm_free_volume_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        sm = attr.binaryVersion;
        }

    // setup the grid to run the kernel
    unsigned int n_groups = min(args.block_size, (unsigned int)max_block_size) / args.group_size / args.stride;

    dim3 threads(args.stride, args.group_size, n_groups);
    dim3 grid( args.n_sample / n_groups + 1, 1, 1);

    // hack to enable grids of more than 65k blocks
    if (sm < 30 && grid.x > 65535)
        {
        grid.y = grid.x / 65535 + 1;
        grid.x = 65535;
        }

    unsigned int shared_bytes = args.num_types * sizeof(typename Shape::param_type) + n_groups*sizeof(unsigned int)
        + args.overlap_idx.getNumElements()*sizeof(unsigned int);

    // required for memory coherency
    cudaDeviceSynchronize();

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - attr.sharedSizeBytes - shared_bytes;

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called
    cudaStreamAttachMemAsync(args.stream, d_params, 0, cudaMemAttachSingle);
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        // attach nested memory regions
        d_params[i].attach_to_stream(args.stream);
        }

    // determine dynamically requested shared memory
    char *ptr = (char *)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        d_params[i].load_shared(ptr, available_bytes);
        }
    unsigned int extra_bytes = max_extra_bytes - available_bytes;

    shared_bytes += extra_bytes;

    gpu_mcm_free_volume_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(
                                                     args.n_sample,
                                                     args.type,
                                                     args.d_postype,
                                                     args.d_orientation,
                                                     args.d_cell_size,
                                                     args.ci,
                                                     args.cli,
                                                     args.d_excell_idx,
                                                     args.d_excell_size,
                                                     args.excli,
                                                     args.cell_dim,
                                                     args.N,
                                                     args.num_types,
                                                     args.seed,
                                                     args.select,
                                                     args.timestep,
                                                     args.dim,
                                                     args.box,
                                                     args.d_n_overlap_all,
                                                     args.ghost_width,
                                                     args.d_check_overlaps,
                                                     args.overlap_idx,
                                                     d_params,
                                                     max_extra_bytes);

    // return control of managed memory
    cudaDeviceSynchronize();

    return cudaSuccess;
    }

#endif // NVCC

}; // end namespace detail

} // end namespace mcm

#endif // _COMPUTE_FREE_VOLUME_CUH_

