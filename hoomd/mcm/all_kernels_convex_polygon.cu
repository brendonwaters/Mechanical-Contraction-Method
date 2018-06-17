// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "IntegratorHPMCMonoImplicitNewGPU.cuh"

#include "ShapeConvexPolygon.h"

namespace mcm
{

namespace detail
{

//! HPMC kernels for ShapeConvexPolygon
template cudaError_t gpu_mcm_free_volume<ShapeConvexPolygon>(const mcm_free_volume_args_t &args,
                                                       const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_mcm_update<ShapeConvexPolygon>(const mcm_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_mcm_implicit_count_overlaps<ShapeConvexPolygon>(const mcm_implicit_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject<ShapeConvexPolygon>(const mcm_implicit_args_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_mcm_insert_depletants_queue<ShapeConvexPolygon>(const mcm_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);
template cudaError_t gpu_mcm_implicit_accept_reject_new<ShapeConvexPolygon>(const mcm_implicit_args_new_t& args,
                                                  const typename ShapeConvexPolygon::param_type *d_params);

}; // end namespace detail

} // end namespace mcm
