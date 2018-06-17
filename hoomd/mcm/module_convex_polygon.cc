// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeConvexPolygon.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"
#include "ExternalCallback.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"
#include "UpdaterClusters.h"
#include "UpdaterClustersImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorMCMMonoGPU.h"
#include "IntegratorMCMMonoImplicitGPU.h"
#include "IntegratorMCMMonoImplicitNewGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif




namespace py = pybind11;
using namespace mcm;

using namespace mcm::detail;

namespace mcm
{

//! Export the base MCMMono integrators
void export_convex_polygon(py::module& m)
    {
    export_IntegratorMCMMono< ShapeConvexPolygon >(m, "IntegratorMCMMonoConvexPolygon");
    export_IntegratorMCMMonoImplicit< ShapeConvexPolygon >(m, "IntegratorMCMMonoImplicitConvexPolygon");
    export_IntegratorMCMMonoImplicitNew< ShapeConvexPolygon >(m, "IntegratorMCMMonoImplicitNewConvexPolygon");
    export_ComputeFreeVolume< ShapeConvexPolygon >(m, "ComputeFreeVolumeConvexPolygon");
    export_AnalyzerSDF< ShapeConvexPolygon >(m, "AnalyzerSDFConvexPolygon");
    export_UpdaterMuVT< ShapeConvexPolygon >(m, "UpdaterMuVTConvexPolygon");
    export_UpdaterMuVTImplicit< ShapeConvexPolygon, IntegratorMCMMonoImplicit<ShapeConvexPolygon> >(m, "UpdaterMuVTImplicitConvexPolygon");
    export_UpdaterMuVTImplicit< ShapeConvexPolygon, IntegratorMCMMonoImplicitNew<ShapeConvexPolygon> >(m, "UpdaterMuVTImplicitNewConvexPolygon");
    export_UpdaterClusters< ShapeConvexPolygon >(m, "UpdaterClustersConvexPolygon");
    export_UpdaterClustersImplicit< ShapeConvexPolygon, IntegratorMCMMonoImplicit<ShapeConvexPolygon> >(m, "UpdaterClustersImplicitConvexPolygon");
    export_UpdaterClustersImplicit< ShapeConvexPolygon, IntegratorMCMMonoImplicitNew<ShapeConvexPolygon> >(m, "UpdaterClustersImplicitNewConvexPolygon");

    export_ExternalFieldInterface<ShapeConvexPolygon>(m, "ExternalFieldConvexPolygon");
    export_LatticeField<ShapeConvexPolygon>(m, "ExternalFieldLatticeConvexPolygon");
    export_ExternalFieldComposite<ShapeConvexPolygon>(m, "ExternalFieldCompositeConvexPolygon");
    export_RemoveDriftUpdater<ShapeConvexPolygon>(m, "RemoveDriftUpdaterConvexPolygon");
    // export_ExternalFieldWall<ShapeConvexPolygon>(m, "WallConvexPolygon");
    // export_UpdaterExternalFieldWall<ShapeConvexPolygon>(m, "UpdaterExternalFieldWallConvexPolygon");
    export_ExternalCallback<ShapeConvexPolygon>(m, "ExternalCallbackConvexPolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapeConvexPolygon >(m, "IntegratorMCMMonoGPUConvexPolygon");
    export_IntegratorMCMMonoImplicitGPU< ShapeConvexPolygon >(m, "IntegratorMCMMonoImplicitGPUConvexPolygon");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeConvexPolygon >(m, "IntegratorMCMMonoImplicitNewGPUConvexPolygon");
    export_ComputeFreeVolumeGPU< ShapeConvexPolygon >(m, "ComputeFreeVolumeGPUConvexPolygon");
    #endif
    }

}
