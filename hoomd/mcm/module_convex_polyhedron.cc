// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeConvexPolyhedron.h"
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
void export_convex_polyhedron(py::module& m)
    {
    export_IntegratorMCMMono< ShapeConvexPolyhedron >(m, "IntegratorMCMMonoConvexPolyhedron");
    export_IntegratorMCMMonoImplicit< ShapeConvexPolyhedron >(m, "IntegratorMCMMonoImplicitConvexPolyhedron");
    export_IntegratorMCMMonoImplicitNew< ShapeConvexPolyhedron >(m, "IntegratorMCMMonoImplicitNewConvexPolyhedron");
    export_ComputeFreeVolume< ShapeConvexPolyhedron >(m, "ComputeFreeVolumeConvexPolyhedron");
    export_AnalyzerSDF< ShapeConvexPolyhedron >(m, "AnalyzerSDFConvexPolyhedron");
    export_UpdaterMuVT< ShapeConvexPolyhedron >(m, "UpdaterMuVTConvexPolyhedron");
    export_UpdaterClusters< ShapeConvexPolyhedron >(m, "UpdaterClustersConvexPolyhedron");
    export_UpdaterClustersImplicit< ShapeConvexPolyhedron, IntegratorMCMMonoImplicit<ShapeConvexPolyhedron> >(m, "UpdaterClustersImplicitConvexPolyhedron");
    export_UpdaterClustersImplicit< ShapeConvexPolyhedron, IntegratorMCMMonoImplicitNew<ShapeConvexPolyhedron> >(m, "UpdaterClustersImplicitNewConvexPolyhedron");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron, IntegratorMCMMonoImplicit<ShapeConvexPolyhedron> >(m, "UpdaterMuVTImplicitConvexPolyhedron");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron, IntegratorMCMMonoImplicitNew<ShapeConvexPolyhedron> >(m, "UpdaterMuVTImplicitNewConvexPolyhedron");

    export_ExternalFieldInterface<ShapeConvexPolyhedron >(m, "ExternalFieldConvexPolyhedron");
    export_LatticeField<ShapeConvexPolyhedron >(m, "ExternalFieldLatticeConvexPolyhedron");
    export_ExternalFieldComposite<ShapeConvexPolyhedron >(m, "ExternalFieldCompositeConvexPolyhedron");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron >(m, "RemoveDriftUpdaterConvexPolyhedron");
    export_ExternalFieldWall<ShapeConvexPolyhedron >(m, "WallConvexPolyhedron");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron >(m, "UpdaterExternalFieldWallConvexPolyhedron");
    export_ExternalCallback<ShapeConvexPolyhedron>(m, "ExternalCallbackConvexPolyhedron");

    #ifdef ENABLE_CUDA

    export_IntegratorMCMMonoGPU< ShapeConvexPolyhedron >(m, "IntegratorMCMMonoGPUConvexPolyhedron");
    export_IntegratorMCMMonoImplicitGPU< ShapeConvexPolyhedron >(m, "IntegratorMCMMonoImplicitGPUConvexPolyhedron");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeConvexPolyhedron >(m, "IntegratorMCMMonoImplicitNewGPUConvexPolyhedron");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron >(m, "ComputeFreeVolumeGPUConvexPolyhedron");

    #endif
    }

}
