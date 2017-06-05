##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2015 Sandia Corporation.
##  Copyright 2015 UT-Battelle, LLC.
##  Copyright 2015 Los Alamos National Security.
##
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
##  Laboratory (LANL), the U.S. Government retains certain rights in
##  this software.
##
##=============================================================================

cmake_minimum_required(VERSION 3.6)

set(VTKm_DIR "/home/users/aom/vtkm_build/lib")
#set(VTK_DIR "/home/users/aom/installs/local/")
#set(VTK_DIR "/home/users/aom/installs/local/include/vtk-6.1")
set(VTK_DIR "/home/users/aom/VTK-build/")

#Find the VTK-m package
find_package(
  VTKm REQUIRED QUIET
  OPTIONAL_COMPONENTS Serial CUDA TBB Rendering
  )

find_package(VTK REQUIRED)

if(VTKm_CUDA_FOUND)
  set(old_nvcc_flags ${CUDA_NVCC_FLAGS})
  set(old_cxx_flags ${CMAKE_CXX_FLAGS})
  vtkm_setup_nvcc_flags( old_nvcc_flags old_cxx_flags)
  vtkm_disable_troublesome_thrust_warnings()

  # Cuda compiles do not respect target_include_directories
  cuda_include_directories(${VTKm_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
  cuda_add_executable(rayCaster rayCaster.cu)

  set(CUDA_NVCC_FLAGS ${old_nvcc_flags})
  set(CMAKE_CXX_FLAGS ${old_cxx_flags})
else()
  add_executable(rayCaster rayCaster.cxx)
endif()

target_include_directories(rayCaster PRIVATE ${VTKm_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
#target_link_libraries(rayCaster PRIVATE ${VTKm_LIBRARIES} ${VTK_LIBRARIES})
if(VTK_LIBRARIES)
    target_link_libraries(rayCaster PRIVATE ${VTKm_LIBRARIES} ${VTK_LIBRARIES})
else()
    target_link_libraries(rayCaster PRIVATE ${VTKm_LIBRARIES} vtkHybrid)
endif()
target_compile_options(rayCaster PRIVATE ${VTKm_COMPILE_OPTIONS})
