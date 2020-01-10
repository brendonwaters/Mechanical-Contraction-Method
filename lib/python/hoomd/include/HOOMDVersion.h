// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __HOOMD_VERSION_H__
#define __HOOMD_VERSION_H__

// Full text HOOMD version string
#define HOOMD_VERSION "2.3.1"
// Full text HOOMD version string
#define HOOMD_VERSION_LONG "v2.3.1-42-gcc47608d4"
// Text string containing the date this build was configured
#define COMPILE_DATE "11/15/2019"
// Set to the src dir to be used as the root to read data files from
#define HOOMD_SOURCE_DIR "/home/brendon/Documents/Research/Mechanical-Contraction-Method"
// Set to the binary dir
#define HOOMD_BINARY_DIR "/home/brendon/Documents/Research/Mechanical-Contraction-Method/build"
// Set the installation directory as a hint for locating libraries
#define HOOMD_INSTALL_PREFIX ""
// Current git refspec checked out
#define HOOMD_GIT_REFSPEC "refs/heads/master"
// Current git hash
#define HOOMD_GIT_SHA1 "cc47608d45eaa16ef47eada9b69236733af3743d"

// hoomd major version
#define HOOMD_VERSION_MAJOR 2
// hoomd minor version
#define HOOMD_VERSION_MINOR 3
// hoomd patch version
#define HOOMD_VERSION_PATCH 1

#include <string>

/*! \file HOOMDVersion.h.in
	\brief Functions and variables for printing compile time build information of HOOMD
	\details This file is configured to HOOMDVersion.h by CMake, that is the file that should
		be included in any code needing it.

	\ingroup utils
*/

//! Formats the HOOMD compile flags as a strong
std::string hoomd_compile_flags();

//! Formats the HOOMD version info header as string
std::string output_version_info();

#endif
