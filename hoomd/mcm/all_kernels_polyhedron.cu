// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorMCMMonoGPU.cuh"
#include "IntegratorMCMMonoImplicitGPU.cuh"
#include "IntegratorMCMMonoImplicitNewGPU.cuh"

#include "ShapePolyhedron.h"

namespace mcm
{

namespace detail
{

//! MCM kernels for ShapePolyhedron
template cudaError_t gpu_mcm_free_volume<ShapePolyhedron>(const mcm_free_volume_args_t &args,
                                                       const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_mcm_update<ShapePolyhedron>(const mcm_args_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_mcm_implicit_count_overlaps<ShapePolyhedron>(const mcm_implicit_args_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject<ShapePolyhedron>(const mcm_implicit_args_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_mcm_insert_depletants_queue<ShapePolyhedron>(const mcm_implicit_args_new_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject_new<ShapePolyhedron>(const mcm_implicit_args_new_t& args,
                                                  const typename ShapePolyhedron::param_type *d_params);

}; // end namespace detail

} // end namespace mcm
