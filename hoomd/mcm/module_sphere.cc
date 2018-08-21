// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeSphere.h"
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
void export_sphere(py::module& m)
    {
    export_IntegratorMCMMono< ShapeSphere >(m, "IntegratorMCMMonoSphere");
    export_IntegratorMCMMonoImplicit< ShapeSphere >(m, "IntegratorMCMMonoImplicitSphere");
    export_IntegratorMCMMonoImplicitNew< ShapeSphere >(m, "IntegratorMCMMonoImplicitNewSphere");
    export_ComputeFreeVolume< ShapeSphere >(m, "ComputeFreeVolumeSphere");
    export_AnalyzerSDF< ShapeSphere >(m, "AnalyzerSDFSphere");
    export_UpdaterMuVT< ShapeSphere >(m, "UpdaterMuVTSphere");
    export_UpdaterClusters< ShapeSphere >(m, "UpdaterClustersSphere");
    export_UpdaterClustersImplicit< ShapeSphere,IntegratorMCMMonoImplicit<ShapeSphere> >(m, "UpdaterClustersImplicitSphere");
    export_UpdaterClustersImplicit< ShapeSphere,IntegratorMCMMonoImplicitNew<ShapeSphere> >(m, "UpdaterClustersImplicitNewSphere");
    export_UpdaterMuVTImplicit< ShapeSphere, IntegratorMCMMonoImplicit<ShapeSphere> >(m, "UpdaterMuVTImplicitSphere");
    export_UpdaterMuVTImplicit< ShapeSphere, IntegratorMCMMonoImplicitNew<ShapeSphere> >(m, "UpdaterMuVTImplicitNewSphere");

    export_ExternalFieldInterface<ShapeSphere>(m, "ExternalFieldSphere");
    export_LatticeField<ShapeSphere>(m, "ExternalFieldLatticeSphere");
    export_ExternalFieldComposite<ShapeSphere>(m, "ExternalFieldCompositeSphere");
    export_RemoveDriftUpdater<ShapeSphere>(m, "RemoveDriftUpdaterSphere");
    export_ExternalFieldWall<ShapeSphere>(m, "WallSphere");
    export_UpdaterExternalFieldWall<ShapeSphere>(m, "UpdaterExternalFieldWallSphere");
    export_ExternalCallback<ShapeSphere>(m, "ExternalCallbackSphere");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapeSphere >(m, "IntegratorMCMMonoGPUSphere");
    export_IntegratorMCMMonoImplicitGPU< ShapeSphere >(m, "IntegratorMCMMonoImplicitGPUSphere");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeSphere >(m, "IntegratorMCMMonoImplicitNewGPUSphere");
    export_ComputeFreeVolumeGPU< ShapeSphere >(m, "ComputeFreeVolumeGPUSphere");
    #endif
    }

}
