// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapePolyhedron.h"
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
void export_polyhedron(py::module& m)
    {
    export_IntegratorMCMMono< ShapePolyhedron >(m, "IntegratorMCMMonoPolyhedron");
    export_IntegratorMCMMonoImplicit< ShapePolyhedron >(m, "IntegratorMCMMonoImplicitPolyhedron");
    export_IntegratorMCMMonoImplicitNew< ShapePolyhedron >(m, "IntegratorMCMMonoImplicitNewPolyhedron");
    export_ComputeFreeVolume< ShapePolyhedron >(m, "ComputeFreeVolumePolyhedron");
    // export_AnalyzerSDF< ShapePolyhedron >(m, "AnalyzerSDFPolyhedron");
    export_UpdaterMuVT< ShapePolyhedron >(m, "UpdaterMuVTPolyhedron");
    export_UpdaterClusters< ShapePolyhedron >(m, "UpdaterClustersPolyhedron");
    export_UpdaterClustersImplicit< ShapePolyhedron, IntegratorMCMMonoImplicit<ShapePolyhedron> >(m, "UpdaterClustersImplicitPolyhedron");
    export_UpdaterClustersImplicit< ShapePolyhedron, IntegratorMCMMonoImplicitNew<ShapePolyhedron> >(m, "UpdaterClustersImplicitNewPolyhedron");
    export_UpdaterMuVTImplicit< ShapePolyhedron, IntegratorMCMMonoImplicit<ShapePolyhedron> >(m, "UpdaterMuVTImplicitPolyhedron");
    export_UpdaterMuVTImplicit< ShapePolyhedron, IntegratorMCMMonoImplicitNew<ShapePolyhedron> >(m, "UpdaterMuVTImplicitNewPolyhedron");

    export_ExternalFieldInterface<ShapePolyhedron>(m, "ExternalFieldPolyhedron");
    export_LatticeField<ShapePolyhedron>(m, "ExternalFieldLatticePolyhedron");
    export_ExternalFieldComposite<ShapePolyhedron>(m, "ExternalFieldCompositePolyhedron");
    export_RemoveDriftUpdater<ShapePolyhedron>(m, "RemoveDriftUpdaterPolyhedron");
    export_ExternalFieldWall<ShapePolyhedron>(m, "WallPolyhedron");
    export_UpdaterExternalFieldWall<ShapePolyhedron>(m, "UpdaterExternalFieldWallPolyhedron");
    export_ExternalCallback<ShapePolyhedron>(m, "ExternalCallbackPolyhedron");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapePolyhedron >(m, "IntegratorMCMMonoGPUPolyhedron");
    export_IntegratorMCMMonoImplicitGPU< ShapePolyhedron >(m, "IntegratorMCMMonoImplicitGPUPolyhedron");
    export_IntegratorMCMMonoImplicitNewGPU< ShapePolyhedron >(m, "IntegratorMCMMonoImplicitNewGPUPolyhedron");
    export_ComputeFreeVolumeGPU< ShapePolyhedron >(m, "ComputeFreeVolumeGPUPolyhedron");
    #endif
    }

}
