// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorMCMMonoGPU.cuh"
#include "IntegratorMCMMonoImplicitGPU.cuh"
#include "IntegratorMCMMonoImplicitNewGPU.cuh"

#include "ShapeSphinx.h"

namespace mcm
{

namespace detail
{
#ifdef ENABLE_SPHINX_GPU
//! MCM kernels for ShapeSphinx
template cudaError_t gpu_mcm_free_volume<ShapeSphinx>(const mcm_free_volume_args_t &args,
                                                       const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_mcm_update<ShapeSphinx>(const mcm_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_mcm_count_overlaps<ShapeSphinx>(const mcm_implicit_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject<ShapeSphinx>(const mcm_implicit_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_mcm_insert_depletants_queue<ShapeSphinx>(const mcm_implicit_args_new_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject_new<ShapeSphinx>(const mcm_implicit_args_new_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
#endif
}; // end namespace detail

} // end namespace mcm
