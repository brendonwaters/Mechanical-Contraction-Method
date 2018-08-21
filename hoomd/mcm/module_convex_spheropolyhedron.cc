// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeSpheropolyhedron.h"
#include "AnalyzerSDF.h"
// #include "ShapeUnion.h"

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
void export_convex_spheropolyhedron(py::module& m)
    {
    export_IntegratorMCMMono< ShapeSpheropolyhedron >(m, "IntegratorMCMMonoSpheropolyhedron");
    export_IntegratorMCMMonoImplicit< ShapeSpheropolyhedron >(m, "IntegratorMCMMonoImplicitSpheropolyhedron");
    export_IntegratorMCMMonoImplicitNew< ShapeSpheropolyhedron >(m, "IntegratorMCMMonoImplicitNewSpheropolyhedron");
    export_ComputeFreeVolume< ShapeSpheropolyhedron >(m, "ComputeFreeVolumeSpheropolyhedron");
    export_AnalyzerSDF< ShapeSpheropolyhedron >(m, "AnalyzerSDFSpheropolyhedron");
    export_UpdaterMuVT< ShapeSpheropolyhedron >(m, "UpdaterMuVTSpheropolyhedron");
    export_UpdaterClusters< ShapeSpheropolyhedron >(m, "UpdaterClustersSpheropolyhedron");
    export_UpdaterClustersImplicit< ShapeSpheropolyhedron, IntegratorMCMMonoImplicit<ShapeSpheropolyhedron> >(m, "UpdaterClustersImplicitSpheropolyhedron");
    export_UpdaterClustersImplicit< ShapeSpheropolyhedron, IntegratorMCMMonoImplicitNew<ShapeSpheropolyhedron> >(m, "UpdaterClustersImplicitNewSpheropolyhedron");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron, IntegratorMCMMonoImplicit<ShapeSpheropolyhedron> >(m, "UpdaterMuVTImplicitSpheropolyhedron");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron, IntegratorMCMMonoImplicitNew<ShapeSpheropolyhedron> >(m, "UpdaterMuVTImplicitNewSpheropolyhedron");

    export_ExternalFieldInterface<ShapeSpheropolyhedron >(m, "ExternalFieldSpheropolyhedron");
    export_LatticeField<ShapeSpheropolyhedron >(m, "ExternalFieldLatticeSpheropolyhedron");
    export_ExternalFieldComposite<ShapeSpheropolyhedron >(m, "ExternalFieldCompositeSpheropolyhedron");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron >(m, "RemoveDriftUpdaterSpheropolyhedron");
    export_ExternalFieldWall<ShapeSpheropolyhedron >(m, "WallSpheropolyhedron");
    export_UpdaterExternalFieldWall<ShapeSpheropolyhedron >(m, "UpdaterExternalFieldWallSpheropolyhedron");
    export_ExternalCallback<ShapeSpheropolyhedron>(m, "ExternalCallbackSpheropolyhedron");

    #ifdef ENABLE_CUDA

    export_IntegratorMCMMonoGPU< ShapeSpheropolyhedron >(m, "IntegratorMCMMonoGPUSpheropolyhedron");
    export_IntegratorMCMMonoImplicitGPU< ShapeSpheropolyhedron >(m, "IntegratorMCMMonoImplicitGPUSpheropolyhedron");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeSpheropolyhedron >(m, "IntegratorMCMMonoImplicitNewGPUSpheropolyhedron");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron >(m, "ComputeFreeVolumeGPUSpheropolyhedron");

    #endif
    }

}
