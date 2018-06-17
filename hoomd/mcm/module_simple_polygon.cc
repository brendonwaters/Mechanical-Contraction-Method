// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeSimplePolygon.h"
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
void export_simple_polygon(py::module& m)
    {
    export_IntegratorMCMMono< ShapeSimplePolygon >(m, "IntegratorMCMMonoSimplePolygon");
    export_IntegratorMCMMonoImplicit< ShapeSimplePolygon >(m, "IntegratorMCMMonoImplicitSimplePolygon");
    export_IntegratorMCMMonoImplicitNew< ShapeSimplePolygon >(m, "IntegratorMCMMonoImplicitNewSimplePolygon");
    export_ComputeFreeVolume< ShapeSimplePolygon >(m, "ComputeFreeVolumeSimplePolygon");
    export_AnalyzerSDF< ShapeSimplePolygon >(m, "AnalyzerSDFSimplePolygon");
    export_UpdaterMuVT< ShapeSimplePolygon >(m, "UpdaterMuVTSimplePolygon");
    export_UpdaterClusters< ShapeSimplePolygon >(m, "UpdaterClustersSimplePolygon");
    export_UpdaterClustersImplicit< ShapeSimplePolygon, IntegratorMCMMonoImplicit<ShapeSimplePolygon> >(m, "UpdaterClustersImplicitSimplePolygon");
    export_UpdaterClustersImplicit< ShapeSimplePolygon, IntegratorMCMMonoImplicitNew<ShapeSimplePolygon> >(m, "UpdaterClustersImplicitNewSimplePolygon");
    export_UpdaterMuVTImplicit< ShapeSimplePolygon, IntegratorMCMMonoImplicit<ShapeSimplePolygon> >(m, "UpdaterMuVTImplicitSimplePolygon");
    export_UpdaterMuVTImplicit< ShapeSimplePolygon, IntegratorMCMMonoImplicitNew<ShapeSimplePolygon> >(m, "UpdaterMuVTImplicitNewSimplePolygon");

    export_ExternalFieldInterface<ShapeSimplePolygon>(m, "ExternalFieldSimplePolygon");
    export_LatticeField<ShapeSimplePolygon>(m, "ExternalFieldLatticeSimplePolygon");
    export_ExternalFieldComposite<ShapeSimplePolygon>(m, "ExternalFieldCompositeSimplePolygon");
    export_RemoveDriftUpdater<ShapeSimplePolygon>(m, "RemoveDriftUpdaterSimplePolygon");
    // export_ExternalFieldWall<ShapeSimplePolygon>(m, "WallSimplePolygon");
    // export_UpdaterExternalFieldWall<ShapeSimplePolygon>(m, "UpdaterExternalFieldWallSimplePolygon");
    export_ExternalCallback<ShapeSimplePolygon>(m, "ExternalCallbackSimplePolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapeSimplePolygon >(m, "IntegratorMCMMonoGPUSimplePolygon");
    export_IntegratorMCMMonoImplicitGPU< ShapeSimplePolygon >(m, "IntegratorMCMMonoImplicitGPUSimplePolygon");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeSimplePolygon >(m, "IntegratorMCMMonoImplicitNewGPUSimplePolygon");
    export_ComputeFreeVolumeGPU< ShapeSimplePolygon >(m, "ComputeFreeVolumeGPUSimplePolygon");
    #endif
    }

}
