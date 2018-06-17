// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"
#include "AnalyzerSDF.h"

#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

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
void export_union_convex_polyhedron(py::module& m)
    {
    export_IntegratorMCMMono< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorMCMMonoConvexPolyhedronUnion");
    export_IntegratorMCMMonoImplicit< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorMCMMonoImplicitConvexPolyhedronUnion");
    export_IntegratorMCMMonoImplicitNew< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorMCMMonoImplicitNewConvexPolyhedronUnion");
    export_ComputeFreeVolume< ShapeUnion<ShapeConvexPolyhedron> >(m, "ComputeFreeVolumeConvexPolyhedronUnion");
    // export_AnalyzerSDF< ShapeUnion<ShapeConvexPolyhedron> >(m, "AnalyzerSDFConvexPolyhedronUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeConvexPolyhedron> >(m, "UpdaterMuVTConvexPolyhedronUnion");
    export_UpdaterClusters<ShapeUnion<ShapeConvexPolyhedron> >(m, "UpdaterClustersConvexPolyhedronUnion");
    export_UpdaterClustersImplicit<ShapeUnion<ShapeConvexPolyhedron>, IntegratorMCMMonoImplicit<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterClustersImplicitConvexPolyhedronUnion");
    export_UpdaterClustersImplicit<ShapeUnion<ShapeConvexPolyhedron>, IntegratorMCMMonoImplicitNew<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterClustersImplicitNewConvexPolyhedronUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeConvexPolyhedron>, IntegratorMCMMonoImplicit<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterMuVTImplicitConvexPolyhedronUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeConvexPolyhedron>, IntegratorMCMMonoImplicitNew<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterMuVTImplicitNewConvexPolyhedronUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeConvexPolyhedron> >(m, "ExternalFieldConvexPolyhedronUnion");
    export_LatticeField<ShapeUnion<ShapeConvexPolyhedron> >(m, "ExternalFieldLatticeConvexPolyhedronUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeConvexPolyhedron> >(m, "ExternalFieldCompositeConvexPolyhedronUnion");
    export_RemoveDriftUpdater<ShapeUnion<ShapeConvexPolyhedron> >(m, "RemoveDriftUpdaterConvexPolyhedronUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron> >(m, "WallConvexPolyhedronUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron> >(m, "UpdaterExternalFieldWallConvexPolyhedronUnion");

    #ifdef ENABLE_CUDA

    export_IntegratorMCMMonoGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorMCMMonoGPUConvexPolyhedronUnion");
    export_IntegratorMCMMonoImplicitGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorMCMMonoImplicitGPUConvexPolyhedronUnion");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorMCMMonoImplicitNewGPUConvexPolyhedronUnion");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "ComputeFreeVolumeGPUConvexPolyhedronUnion");

    #endif
    }

}
