# Maintainer: joaander

option(ENABLE_MCM_SPHINX_GPU "Enable sphinx on the GPU" OFF)
if (ENABLE_MCM_SPHINX_GPU)
    add_definitions(-DENABLE_SPHINX_GPU)
endif()

set(_mcm_sources   module.cc
                    module_external_field.cc
                    module_sphere.cc
                    module_convex_polygon.cc
                    module_spheropolygon.cc
                    module_convex_polyhedron.cc
                    module_convex_spheropolyhedron.cc
                    UpdaterBoxMC.cc
                    IntegratorMCM.cc
                    )

set(_mcm_headers
    AnalyzerSDF.h
    ComputeFreeVolumeGPU.h
    ComputeFreeVolume.h
    ExternalFieldComposite.h
    ExternalField.h
    ExternalFieldLattice.h
    ExternalFieldWall.h
    GPUTree.h
    MCMCounters.h
    MCMPrecisionSetup.h
    IntegratorMCM.h
    IntegratorMCMMonoGPU.h
    IntegratorMCMMono.h
    IntegratorMCMMonoImplicitGPU.h
    IntegratorMCMMonoImplicit.h
    IntegratorMCMMonoImplicitNew.h
    IntegratorMCMMonoImplicitNewGPU.h
    MinkowskiMath.h
    modules.h
    Moves.h
    OBB.h
    OBBTree.h
    ShapeConvexPolygon.h
    ShapeConvexPolyhedron.h
    ShapeSphere.h
    ShapeSpheropolygon.h
    ShapeSpheropolyhedron.h
    ShapeProxy.h
    SphinxOverlap.h
    UpdaterClusters.h
    UpdaterExternalFieldWall.h
    UpdaterMuVT.h
    UpdaterMuVTImplicit.h
    UpdaterRemoveDrift.h
    XenoCollide2D.h
    XenoCollide3D.h
    )

# if (ENABLE_CUDA)
# list(APPEND _mcm_sources
#                            )
# endif()

set(_mcm_cu_sources IntegratorMCMMonoGPU.cu
                     IntegratorMCMMonoImplicitGPU.cu
                     all_kernels_sphere.cu
                     all_kernels_convex_polygon.cu
                     all_kernels_simple_polygon.cu
                     all_kernels_spheropolygon.cu
                     all_kernels_polyhedron.cu
                     all_kernels_ellipsoid.cu
                     all_kernels_faceted_sphere.cu
                     all_kernels_sphinx.cu
                     all_kernels_union_convex_polyhedron.cu
                     all_kernels_union_sphere.cu
                     all_kernels_convex_polyhedron.cu
                     all_kernels_convex_spheropolyhedron.cu
                     )

# quiet some warnings locally on files we can't modify
if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties(IntegratorMCMMonoImplicitGPU.cu PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
endif()

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_mcm_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module (_mcm SHARED ${_mcm_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)
set_target_properties(_mcm PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${PYTHON_MODULE_BASE_DIR})

# link the library to its dependencies
target_link_libraries(_mcm PRIVATE ${HOOMD_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_mcm PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_mcm PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

fix_cudart_rpath(_mcm)

# install the library
install(TARGETS _mcm
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/mcm
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/mcm/${file}"
    )
ENDMACRO(copy_file)

set(files   analyze.py
            compute.py
            data.py
            __init__.py
            integrate.py
            update.py
            util.py
            field.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/mcm
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_mcm ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/mcm/${file}
        COMMENT    "Copy hoomd/include/hoomd/mcm/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_mcm_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/mcm
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_mcm_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_mcm_headers ALL DEPENDS ${_mcm_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

if (BUILD_VALIDATION)
    add_subdirectory(validation)
endif()
