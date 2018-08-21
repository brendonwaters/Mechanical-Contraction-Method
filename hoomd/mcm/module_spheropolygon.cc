// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeSpheropolygon.h"
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
void export_spheropolygon(py::module& m)
    {
    export_IntegratorMCMMono< ShapeSpheropolygon >(m, "IntegratorMCMMonoSpheropolygon");
    export_IntegratorMCMMonoImplicit< ShapeSpheropolygon >(m, "IntegratorMCMMonoImplicitSpheropolygon");
    export_IntegratorMCMMonoImplicitNew< ShapeSpheropolygon >(m, "IntegratorMCMMonoImplicitNewSpheropolygon");
    export_ComputeFreeVolume< ShapeSpheropolygon >(m, "ComputeFreeVolumeSpheropolygon");
    export_AnalyzerSDF< ShapeSpheropolygon >(m, "AnalyzerSDFSpheropolygon");
    export_UpdaterMuVT< ShapeSpheropolygon >(m, "UpdaterMuVTSpheropolygon");
    export_UpdaterClusters< ShapeSpheropolygon >(m, "UpdaterClustersSpheropolygon");
    export_UpdaterClustersImplicit< ShapeSpheropolygon, IntegratorMCMMonoImplicit<ShapeSpheropolygon> >(m, "UpdaterClustersImplicitSpheropolygon");
    export_UpdaterClustersImplicit< ShapeSpheropolygon, IntegratorMCMMonoImplicitNew<ShapeSpheropolygon> >(m, "UpdaterClustersImplicitNewSpheropolygon");
    export_UpdaterMuVTImplicit< ShapeSpheropolygon, IntegratorMCMMonoImplicit<ShapeSpheropolygon> >(m, "UpdaterMuVTImplicitSpheropolygon");
    export_UpdaterMuVTImplicit< ShapeSpheropolygon, IntegratorMCMMonoImplicitNew<ShapeSpheropolygon> >(m, "UpdaterMuVTImplicitNewSpheropolygon");

    export_ExternalFieldInterface<ShapeSpheropolygon>(m, "ExternalFieldSpheropolygon");
    export_LatticeField<ShapeSpheropolygon>(m, "ExternalFieldLatticeSpheropolygon");
    export_ExternalFieldComposite<ShapeSpheropolygon>(m, "ExternalFieldCompositeSpheropolygon");
    export_RemoveDriftUpdater<ShapeSpheropolygon>(m, "RemoveDriftUpdaterSpheropolygon");
    // export_ExternalFieldWall<ShapeSpheropolygon>(m, "WallSpheropolygon");
    // export_UpdaterExternalFieldWall<ShapeSpheropolygon>(m, "UpdaterExternalFieldWallSpheropolygon");
    export_ExternalCallback<ShapeSpheropolygon>(m, "ExternalCallbackSpheropolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapeSpheropolygon >(m, "IntegratorMCMMonoGPUSpheropolygon");
    export_IntegratorMCMMonoImplicitGPU< ShapeSpheropolygon >(m, "IntegratorMCMMonoImplicitGPUSpheropolygon");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeSpheropolygon >(m, "IntegratorMCMMonoImplicitNewGPUSpheropolygon");
    export_ComputeFreeVolumeGPU< ShapeSpheropolygon >(m, "ComputeFreeVolumeGPUSpheropolygon");
    #endif
    }

}
