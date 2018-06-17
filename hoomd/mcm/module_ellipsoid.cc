// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeEllipsoid.h"
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
void export_ellipsoid(py::module& m)
    {
    export_IntegratorMCMMono< ShapeEllipsoid >(m, "IntegratorMCMMonoEllipsoid");
    export_IntegratorMCMMonoImplicit< ShapeEllipsoid >(m, "IntegratorMCMMonoImplicitEllipsoid");
    export_IntegratorMCMMonoImplicitNew< ShapeEllipsoid >(m, "IntegratorMCMMonoImplicitNewEllipsoid");
    export_ComputeFreeVolume< ShapeEllipsoid >(m, "ComputeFreeVolumeEllipsoid");
    export_AnalyzerSDF< ShapeEllipsoid >(m, "AnalyzerSDFEllipsoid");
    export_UpdaterMuVT< ShapeEllipsoid >(m, "UpdaterMuVTEllipsoid");
    export_UpdaterClusters< ShapeEllipsoid >(m, "UpdaterClustersEllipsoid");
    export_UpdaterClustersImplicit< ShapeEllipsoid, IntegratorMCMMonoImplicit<ShapeEllipsoid> >(m, "UpdaterClustersImplicitEllipsoid");
    export_UpdaterClustersImplicit< ShapeEllipsoid, IntegratorMCMMonoImplicitNew<ShapeEllipsoid> >(m, "UpdaterClustersImplicitNewEllipsoid");
    export_UpdaterMuVTImplicit< ShapeEllipsoid, IntegratorMCMMonoImplicit<ShapeEllipsoid> >(m, "UpdaterMuVTImplicitEllipsoid");
    export_UpdaterMuVTImplicit< ShapeEllipsoid, IntegratorMCMMonoImplicitNew<ShapeEllipsoid> >(m, "UpdaterMuVTImplicitNewEllipsoid");

    export_ExternalFieldInterface<ShapeEllipsoid>(m, "ExternalFieldEllipsoid");
    export_LatticeField<ShapeEllipsoid>(m, "ExternalFieldLatticeEllipsoid");
    export_ExternalFieldComposite<ShapeEllipsoid>(m, "ExternalFieldCompositeEllipsoid");
    export_RemoveDriftUpdater<ShapeEllipsoid>(m, "RemoveDriftUpdaterEllipsoid");
    export_ExternalFieldWall<ShapeEllipsoid>(m, "WallEllipsoid");
    export_UpdaterExternalFieldWall<ShapeEllipsoid>(m, "UpdaterExternalFieldWallEllipsoid");
    export_ExternalCallback<ShapeEllipsoid>(m, "ExternalCallbackEllipsoid");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapeEllipsoid >(m, "IntegratorMCMMonoGPUEllipsoid");
    export_IntegratorMCMMonoImplicitGPU< ShapeEllipsoid >(m, "IntegratorMCMMonoImplicitGPUEllipsoid");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeEllipsoid >(m, "IntegratorMCMMonoImplicitNewGPUEllipsoid");
    export_ComputeFreeVolumeGPU< ShapeEllipsoid >(m, "ComputeFreeVolumeGPUEllipsoid");
    #endif
    }

}
