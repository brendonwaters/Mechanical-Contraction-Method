// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorMCM.h"
#include "IntegratorMCMMono.h"
#include "IntegratorMCMMonoImplicit.h"
#include "IntegratorMCMMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapeSphinx.h"
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
void export_sphinx(py::module& m)
    {
    export_IntegratorMCMMono< ShapeSphinx >(m, "IntegratorMCMMonoSphinx");
    export_IntegratorMCMMonoImplicit< ShapeSphinx >(m, "IntegratorMCMMonoImplicitSphinx");
    export_IntegratorMCMMonoImplicitNew< ShapeSphinx >(m, "IntegratorMCMMonoImplicitNewSphinx");
    export_ComputeFreeVolume< ShapeSphinx >(m, "ComputeFreeVolumeSphinx");
    export_AnalyzerSDF< ShapeSphinx >(m, "AnalyzerSDFSphinx");
    export_UpdaterMuVT< ShapeSphinx >(m, "UpdaterMuVTSphinx");
    export_UpdaterClusters< ShapeSphinx >(m, "UpdaterClustersSphinx");
    export_UpdaterClustersImplicit< ShapeSphinx, IntegratorMCMMonoImplicit<ShapeSphinx> >(m, "UpdaterClustersImplicitSphinx");
    export_UpdaterClustersImplicit< ShapeSphinx, IntegratorMCMMonoImplicitNew<ShapeSphinx> >(m, "UpdaterClustersImplicitNewSphinx");
    export_UpdaterMuVTImplicit< ShapeSphinx, IntegratorMCMMonoImplicit<ShapeSphinx> >(m, "UpdaterMuVTImplicitSphinx");
    export_UpdaterMuVTImplicit< ShapeSphinx, IntegratorMCMMonoImplicitNew<ShapeSphinx> >(m, "UpdaterMuVTImplicitNewSphinx");

    export_ExternalFieldInterface<ShapeSphinx>(m, "ExternalFieldSphinx");
    export_LatticeField<ShapeSphinx>(m, "ExternalFieldLatticeSphinx");
    export_ExternalFieldComposite<ShapeSphinx>(m, "ExternalFieldCompositeSphinx");
    export_RemoveDriftUpdater<ShapeSphinx>(m, "RemoveDriftUpdaterSphinx");
    export_ExternalFieldWall<ShapeSphinx>(m, "WallSphinx");
    export_UpdaterExternalFieldWall<ShapeSphinx>(m, "UpdaterExternalFieldWallSphinx");
    export_ExternalCallback<ShapeSphinx>(m, "ExternalCallbackSphinx");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorMCMMonoGPU< ShapeSphinx >(m, "IntegratorMCMMonoGPUSphinx");
    export_IntegratorMCMMonoImplicitGPU< ShapeSphinx >(m, "IntegratorMCMMonoImplicitGPUSphinx");
    export_IntegratorMCMMonoImplicitNewGPU< ShapeSphinx >(m, "IntegratorMCMMonoImplicitNewGPUSphinx");
    export_ComputeFreeVolumeGPU< ShapeSphinx >(m, "ComputeFreeVolumeGPUSphinx");

    #endif
    #endif
    }

}
