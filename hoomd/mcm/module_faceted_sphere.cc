// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeFacetedSphere.h"
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
void export_faceted_sphere(py::module& m)
    {
    export_IntegratorMCMMono< ShapeFacetedSphere >(m, "IntegratorMCMMonoFacetedSphere");
    export_IntegratorMCMMonoImplicit< ShapeFacetedSphere >(m, "IntegratorMCMMonoImplicitFacetedSphere");
    export_IntegratorMCMMonoImplicitNew< ShapeFacetedSphere >(m, "IntegratorMCMMonoImplicitNewFacetedSphere");
    export_ComputeFreeVolume< ShapeFacetedSphere >(m, "ComputeFreeVolumeFacetedSphere");
    export_AnalyzerSDF< ShapeFacetedSphere >(m, "AnalyzerSDFFacetedSphere");
    export_UpdaterMuVT< ShapeFacetedSphere >(m, "UpdaterMuVTFacetedSphere");
    export_UpdaterClusters< ShapeFacetedSphere >(m, "UpdaterClustersFacetedSphere");
    export_UpdaterClustersImplicit< ShapeFacetedSphere, IntegratorMCMMonoImplicit<ShapeFacetedSphere> >(m, "UpdaterClustersImplicitFacetedSphere");
    export_UpdaterClustersImplicit< ShapeFacetedSphere, IntegratorMCMMonoImplicitNew<ShapeFacetedSphere> >(m, "UpdaterClustersImplicitNewFacetedSphere");
    export_UpdaterMuVTImplicit< ShapeFacetedSphere, IntegratorMCMMonoImplicit<ShapeFacetedSphere> >(m, "UpdaterMuVTImplicitFacetedSphere");
    export_UpdaterMuVTImplicit< ShapeFacetedSphere, IntegratorMCMMonoImplicitNew<ShapeFacetedSphere> >(m, "UpdaterMuVTImplicitNewFacetedSphere");

    export_ExternalFieldInterface<ShapeFacetedSphere>(m, "ExternalFieldFacetedSphere");
    export_LatticeField<ShapeFacetedSphere>(m, "ExternalFieldLatticeFacetedSphere");
    export_ExternalFieldComposite<ShapeFacetedSphere>(m, "ExternalFieldCompositeFacetedSphere");
    export_RemoveDriftUpdater<ShapeFacetedSphere>(m, "RemoveDriftUpdaterFacetedSphere");
    export_ExternalFieldWall<ShapeFacetedSphere>(m, "WallFacetedSphere");
    export_UpdaterExternalFieldWall<ShapeFacetedSphere>(m, "UpdaterExternalFieldWallFacetedSphere");
    export_ExternalCallback<ShapeFacetedSphere>(m, "ExternalCallbackFacetedSphere");

    #ifdef ENABLE_CUDA
    export_IntegratorMCMMonoGPU< ShapeFacetedSphere >(m, "IntegratorMCMMonoGPUFacetedSphere");
    export_IntegratorMCMMonoImplicitGPU< ShapeFacetedSphere >(m, "IntegratorMCMMonoImplicitGPUFacetedSphere");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeFacetedSphere >(m, "IntegratorMCMMonoImplicitNewGPUFacetedSphere");
    export_ComputeFreeVolumeGPU< ShapeFacetedSphere >(m, "ComputeFreeVolumeGPUFacetedSphere");
    #endif
    }

}
