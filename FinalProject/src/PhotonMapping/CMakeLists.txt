
#
#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

if(GLUT_FOUND AND OPENGL_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
  add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

  set(target_name
    PhotonMapping
    )
  set(host_files
    main.cpp
    photon_mapping_scene.h
    photon_mapping_scene.cpp
    )
  set(optix_files
    ray_tracing.cu
    photon_tracing.cu
    gathering.cu
    output.cu
    parallelogram.cu
    sphere.cu
    inlines.h
    structs.h
    kdtree.h
    select.h
    random.h
    )
  set(cudac_files
    device_functions.h
    device_functions.cu
    new_knn.h
    )

  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  set(CUDA_64_BIT_DEVICE_CODE ON)

  CUDA_WRAP_SRCS(${target_name} PTX ptx_generated_files ${optix_files})

  CUDA_WRAP_SRCS(${target_name} OBJ obj_generated_files ${cudac_files})

  add_executable(${target_name}
    ${host_files}
    ${optix_files}
    ${cudac_files}
    ${ptx_generated_files}
    ${obj_generated_files}
    )

  target_link_libraries(${target_name}
    sutil
    optix
    ${optix_rpath}
    ${CUDA_CUDART_LIBRARY}
    gpuknn
    cudpp_x86_64
    cutil  # must be after cudpp_x86_64
    )

else()
  # GLUT or OpenGL not found
  message("Disabling PhotonMapping, which requires glut and opengl.")
endif()

