// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorMCMMonoGPU.cuh"
#include "IntegratorMCMMonoImplicitGPU.cuh"
#include "IntegratorMCMMonoImplicitNewGPU.cuh"

#include "ShapeUnion.h"

namespace mcm
{

namespace detail
{

//! MCM kernels for ShapeUnion<ShapeSphere>
template cudaError_t gpu_mcm_free_volume<ShapeUnion<ShapeSphere> >(const mcm_free_volume_args_t &args,
                                                       const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_mcm_update<ShapeUnion<ShapeSphere> >(const mcm_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_mcm_implicit_count_overlaps<ShapeUnion<ShapeSphere> >(const mcm_implicit_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject<ShapeUnion<ShapeSphere> >(const mcm_implicit_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_mcm_insert_depletants_queue<ShapeUnion<ShapeSphere> >(const mcm_implicit_args_new_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject_new<ShapeUnion<ShapeSphere> >(const mcm_implicit_args_new_t& args,
                                                  const typename ShapeUnion<ShapeSphere> ::param_type *d_params);

}; // end namespace detail

} // end namespace mcm
