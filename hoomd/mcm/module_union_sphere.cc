// Copyright (c) 2009-206 The Regents of the University of Michigan
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
void export_union_sphere(py::module& m)
    {
    export_IntegratorMCMMono< ShapeUnion<ShapeSphere> >(m, "IntegratorMCMMonoSphereUnion");
    export_IntegratorMCMMonoImplicit< ShapeUnion<ShapeSphere> >(m, "IntegratorMCMMonoImplicitSphereUnion");
    export_IntegratorMCMMonoImplicitNew< ShapeUnion<ShapeSphere> >(m, "IntegratorMCMMonoImplicitNewSphereUnion");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere> >(m, "ComputeFreeVolumeSphereUnion");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, , > >(m, "AnalyzerSDFSphereUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere> >(m, "UpdaterMuVTSphereUnion");
    export_UpdaterClusters< ShapeUnion<ShapeSphere> >(m, "UpdaterClustersSphereUnion");
    export_UpdaterClustersImplicit< ShapeUnion<ShapeSphere>, IntegratorMCMMonoImplicit<ShapeUnion<ShapeSphere> > >(m, "UpdaterClustersImplicitSphereUnion");
    export_UpdaterClustersImplicit< ShapeUnion<ShapeSphere>, IntegratorMCMMonoImplicitNew<ShapeUnion<ShapeSphere> > >(m, "UpdaterClustersImplicitNewSphereUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere>, IntegratorMCMMonoImplicit<ShapeUnion<ShapeSphere> > >(m, "UpdaterMuVTImplicitSphereUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere>, IntegratorMCMMonoImplicitNew<ShapeUnion<ShapeSphere> > >(m, "UpdaterMuVTImplicitNewSphereUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere> >(m, "ExternalFieldSphereUnion");
    export_LatticeField<ShapeUnion<ShapeSphere> >(m, "ExternalFieldLatticeSphereUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere> >(m, "ExternalFieldCompositeSphereUnion");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere> >(m, "RemoveDriftUpdaterSphereUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere> >(m, "WallSphereUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere> >(m, "UpdaterExternalFieldWallSphereUnion");
    export_ExternalCallback<ShapeUnion<ShapeSphere> >(m, "ExternalCallbackSphereUnion");

    #ifdef ENABLE_CUDA

    export_IntegratorMCMMonoGPU< ShapeUnion<ShapeSphere> >(m, "IntegratorMCMMonoGPUSphereUnion");
    export_IntegratorMCMMonoImplicitGPU< ShapeUnion<ShapeSphere> >(m, "IntegratorMCMMonoImplicitGPUSphereUnion");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeUnion<ShapeSphere> >(m, "IntegratorMCMMonoImplicitNewGPUSphereUnion");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere> >(m, "ComputeFreeVolumeGPUSphereUnion");

    #endif
    }

}
