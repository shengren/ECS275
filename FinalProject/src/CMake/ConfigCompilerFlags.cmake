
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

# This will create a set of default compiler flags based on the system
# and compiler supplied.

##############################################################
## Compiler libraries
##############################################################

# Some compilers in non default locations have libraries they need in
# order to run properly.  You could change your LD_LIBRARY_PATH or you
# could add the path toth to the rpath of the library or executable.
# This helps with that.

SET(COMPILER_LIBRARY_PATH "${COMPILER_LIBRARY_PATH}" CACHE PATH "Path to compiler libraries" FORCE)
MARK_AS_ADVANCED(COMPILER_LIBRARY_PATH)

IF(EXISTS "${COMPILER_LIBRARY_PATH}")
  SET(rpath_arg "-Wl,-rpath,\"${COMPILER_LIBRARY_PATH}\"")
  # TODO(bigler): remove the old path if there is one
  FORCE_ADD_FLAGS(CMAKE_EXE_LINKER_FLAGS ${rpath_arg})
  FORCE_ADD_FLAGS(CMAKE_MODULE_LINKER_FLAGS ${rpath_arg})
  FORCE_ADD_FLAGS(CMAKE_SHARED_LINKER_FLAGS ${rpath_arg})
ELSE(EXISTS "${COMPILER_LIBRARY_PATH}")
  IF(COMPILER_LIBRARY_PATH)
    MESSAGE(FATAL_ERROR "COMPILER_LIBRARY_PATH is set, but the path does not exist:\n${COMPILER_LIBRARY_PATH}")
  ENDIF(COMPILER_LIBRARY_PATH)
ENDIF(EXISTS "${COMPILER_LIBRARY_PATH}")

##############################################################
## Helper macros
##############################################################

macro(set_flags FLAG NEW_VALUE)
  if(${NEW_VALUE})
#     first_time_message("Setting compiler flags:")
#     first_time_message("${NEW_VALUE} = ${${NEW_VALUE}}")
    first_time_set(${FLAG} "${${NEW_VALUE}}" CACHE STRING "Default compiler flags" FORCE)
  endif()
endmacro()

# Appends ${new} to the string of flags in ${flag}_INIT, then uses that variable to set
# ${flags} via the set_flags macro.  Note that ${flag}_INIT isn't modified outside of the
# function's scope.
function(append_and_set flag new)
  APPEND_TO_STRING(${flag}_INIT  "${new}")
  set_flags(${flag} ${flag}_INIT)
endfunction()

##############################################################
## System independent
##############################################################

# Initialize these parameters
SET(C_FLAGS "")
SET(C_FLAGS_DEBUG "")
SET(C_FLAGS_RELEASE "")

SET(CXX_FLAGS "")
SET(CXX_FLAGS_DEBUG "")
SET(CXX_FLAGS_RELEASE "")

SET(INTEL_OPT " ")
SET(GCC_OPT " ")
SET(CL_OPT " ")

# Set some defaults.  CMake provides some defaults in the INIT
# versions of the variables.
APPEND_TO_STRING(C_FLAGS "${CMAKE_C_FLAGS_INIT}")
#
APPEND_TO_STRING(CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT}")

# We don't necessarily want to enable aggressive warnings for everyone.  This can only be
# configured from the command line or from the caller's CMakeLists.txt.
if(NOT DEFINED OPTIX_USE_AGGRESSIVE_WARNINGS)
  set(OPTIX_USE_AGGRESSIVE_WARNINGS OFF)
endif()

#############################################################
# Set the default warning levels for each compiler.  Where the compiler runs on
# multiple architectures, the flags are architecture independent.
#############################################################

if(OPTIX_USE_AGGRESSIVE_WARNINGS)
  SET(CXX_WARNING_FLAGS "-Wall -Werror -Wsign-compare")
  SET(C_WARNING_FLAGS   "${CXX_WARNING_FLAGS} -Wstrict-prototypes -Wdeclaration-after-statement")
else()
  set(C_WARNING_FLAGS   " ")
  set(CXX_WARNING_FLAGS " ")
endif()
SET(DEBUG_FLAGS "-O0 -g3")

# We might consider adding -ffast-math.

SET(RELEASE_FLAGS "-O3 -DNDEBUG -g3 -funroll-loops")
IF   (USING_GCC)
  APPEND_TO_STRING(C_FLAGS         ${C_WARNING_FLAGS})
  APPEND_TO_STRING(C_FLAGS_DEBUG   ${DEBUG_FLAGS})
  APPEND_TO_STRING(C_FLAGS_RELEASE ${RELEASE_FLAGS})
ENDIF(USING_GCC)

IF   (USING_GPP)
  APPEND_TO_STRING(CXX_FLAGS         ${CXX_WARNING_FLAGS})
  APPEND_TO_STRING(CXX_FLAGS_DEBUG   "${DEBUG_FLAGS}")
  APPEND_TO_STRING(CXX_FLAGS_RELEASE ${RELEASE_FLAGS})
ENDIF(USING_GPP)

########################
# Windows flags

# /W3 - more warnings
# /WX - warnings as erros
# /Wp64 - warn about 64 bit issues
# /wd4100 - unreferenced formal parameter
# /wd4305 -
# /wd4355 - 'this' used in initializer list
# /wd4996 - strncpy and other functions are unsafe
# removed /Wp64 flag because it is not reliable (false positives)
# in VS2008 this flag is also deprecated by default
#
# Turn on warnings for level /W3 (/w3XXXX).
# /w34101 - unreference local variable
# /w34189 - local variable is initialized but not referenced
if(OPTIX_USE_AGGRESSIVE_WARNINGS)
  set(WARNING_FLAGS "/WX /wd4355 /wd4996 /w34101 /w34189")
else()
  set(WARNING_FLAGS "    /wd4355 /wd4996")
endif()
SET(DEBUG_FLAGS "")

# Add /MP to get file-level compilation parallelism
SET(PARALLEL_COMPILE_FLAGS /MP)

IF (USING_WINDOWS_CL)
  APPEND_TO_STRING(C_FLAGS       "${PARALLEL_COMPILE_FLAGS} ${WARNING_FLAGS}")
  APPEND_TO_STRING(CXX_FLAGS     "${PARALLEL_COMPILE_FLAGS} ${WARNING_FLAGS}")

  # /Ox       - Full Optimization (should supperseed the /O2 optimization
  # /Ot       - Favor fast code over small code
  # /GL (not used) - Enable link time code generation
  # /arch:SSE - Enable SSE instructions (only for 32 bit builds)
  # /fp:fast  - Use Fast floating point model
  string(REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE_INIT "${CMAKE_C_FLAGS_RELEASE_INIT}")
  string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT}")
  set(CL_OPT "/Ot /fp:fast")
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    APPEND_TO_STRING(CL_OPT "/arch:SSE")
  endif()

  append_and_set(CMAKE_C_FLAGS_RELEASE "${CL_OPT}")
  append_and_set(CMAKE_CXX_FLAGS_RELEASE "${CL_OPT}")
  # Turn these on if you turn /GL on
  # append_and_set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/ltcg")
  # append_and_set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "/ltcg")
  # append_and_set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/ltcg")
ENDIF(USING_WINDOWS_CL)

SET(WARNING_FLAGS "/D_CRT_SECURE_NO_DEPRECATE=1 /Qstd=c99")
IF (USING_WINDOWS_ICL)
  # These are the warnings
  APPEND_TO_STRING(C_FLAGS         ${WARNING_FLAGS})
  APPEND_TO_STRING(CXX_FLAGS         ${WARNING_FLAGS})
ENDIF(USING_WINDOWS_ICL)

##############################################################
## Apple
##############################################################
IF(APPLE)
  APPEND_TO_STRING(GCC_ARCH "nocona")
  APPEND_TO_STRING(GCC_ARCH "prescott")
  APPEND_TO_STRING(GCC_OPT "-msse -msse2 -msse3 -mfpmath=sse")
ENDIF(APPLE)

##############################################################
## X86
##############################################################

# On apple machines CMAKE_SYSTEM_PROCESSOR return i386.

IF (CMAKE_SYSTEM_PROCESSOR MATCHES "i686" OR
    CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  APPEND_TO_STRING(GCC_OPT "-msse -msse2 -msse3 -mfpmath=sse")

  # mtune options

  INCLUDE(LinuxCPUInfo)

  # AMD
  IF(VENDOR_ID MATCHES "AuthenticAMD")
    APPEND_TO_STRING(GCC_ARCH "opteron") # supports 64 bit instructions
    APPEND_TO_STRING(GCC_ARCH "athlon-xp") # no support for 64 bit instructions
    APPEND_TO_STRING(INTEL_OPT "-xW -unroll4")
  ENDIF(VENDOR_ID MATCHES "AuthenticAMD")

  # Intel
  IF(VENDOR_ID MATCHES "GenuineIntel")

    IF(CPU_FAMILY EQUAL 6)

      IF(MODEL EQUAL 15) # (F)

        # This is likely a Core 2
        # APPEND_TO_STRING(GCC_ARCH "kentsfield") # QX6700
        APPEND_TO_STRING(GCC_ARCH "nocona")
        APPEND_TO_STRING(GCC_ARCH "prescott")

        # -xT  Intel(R) Core(TM)2 Duo processors, Intel(R) Core(TM)2 Quad
        # processors, and Intel(R) Xeon(R) processors with SSSE3
        APPEND_TO_STRING(INTEL_OPT "-xT -unroll4")

      ENDIF(MODEL EQUAL 15)

      IF(MODEL EQUAL 14) # (E)
        # This is likely a Core Single or Core Duo.  This doesn't
        # support EM64T.
        APPEND_TO_STRING(GCC_ARCH "prescott")
      ENDIF(MODEL EQUAL 14)
      IF(MODEL LESS 14) #(0-D)
        # This is likely a Pentium3, Pentium M.  Some pentium 3s don't
        # support sse2, in that case fall back to the i686 code.
        APPEND_TO_STRING(GCC_ARCH "pentium-m")
        APPEND_TO_STRING(INTEL_OPT "-xB")
      ENDIF(MODEL LESS 14)
    ENDIF(CPU_FAMILY EQUAL 6)
    IF(CPU_FAMILY EQUAL 15)
      # These are your Pentium 4 and friends
      IF(FLAGS MATCHES "em64t")
        APPEND_TO_STRING(GCC_ARCH "nocona")
        APPEND_TO_STRING(GCC_ARCH "prescott")
      ENDIF(FLAGS MATCHES "em64t")
      APPEND_TO_STRING(GCC_ARCH "pentium4")
      APPEND_TO_STRING(INTEL_OPT "-xP -unroll4 -msse3")
    ENDIF(CPU_FAMILY EQUAL 15)
  ENDIF(VENDOR_ID MATCHES "GenuineIntel")
  APPEND_TO_STRING(GCC_ARCH "i686")

  ###########################################################
  # Some x86_64 specific stuff
  IF   (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    APPEND_TO_STRING(INTEL_OPT "")
  ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  ###########################################################

ENDIF (CMAKE_SYSTEM_PROCESSOR MATCHES "i686" OR
       CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")

##############################################################
## Configure Architecture
##############################################################

# Cycle through the GCC_ARCH args and see which one will pass first.
# Guard this evaluation with PASSED_FIRST_CONFIGURE, to make sure it
# is only done the first time.
IF(USING_GCC OR USING_GPP AND NOT PASSED_FIRST_CONFIGURE)
  SEPARATE_ARGUMENTS(GCC_ARCH)
  # Change the extension based of if we are using both gcc and g++.
  IF(USING_GCC)
    SET(EXTENSION "c")
  ELSE(USING_GCC)
    SET(EXTENSION "cc")
  ENDIF(USING_GCC)
  SET(COMPILE_TEST_SOURCE ${CMAKE_BINARY_DIR}/test/compile-test.${EXTENSION})
  GET_FILENAME_COMPONENT(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  CONFIGURE_FILE("${CMAKE_CURRENT_LIST_DIR}/testmain.c"
    ${COMPILE_TEST_SOURCE} IMMEDIATE COPYONLY)
  FOREACH(ARCH ${GCC_ARCH})
    IF(NOT GOOD_ARCH)
#       MESSAGE("Testing ARCH = ${ARCH}")
      SET(ARCH_FLAG "-march=${ARCH} -mtune=${ARCH}")
      SET(COMPILER_ARGS "${ARCH_FLAG} ${C_FLAGS_RELEASE} ${C_FLAGS} ${GCC_OPT}")
      TRY_RUN(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}/test ${COMPILE_TEST_SOURCE}
        CMAKE_FLAGS
        -DCOMPILE_DEFINITIONS:STRING=${COMPILER_ARGS}
        OUTPUT_VARIABLE OUTPUT
        )
#       MESSAGE("OUTPUT             = ${OUTPUT}")
#       MESSAGE("COMPILER_ARGS      = ${COMPILER_ARGS}")
#       MESSAGE("RUN_RESULT_VAR     = ${RUN_RESULT_VAR}")
#       MESSAGE("COMPILE_RESULT_VAR = ${COMPILE_RESULT_VAR}")
      IF(RUN_RESULT_VAR EQUAL 0)
        SET(GOOD_ARCH ${ARCH})
      ENDIF(RUN_RESULT_VAR EQUAL 0)
    ENDIF(NOT GOOD_ARCH)
  ENDFOREACH(ARCH)
  IF(GOOD_ARCH)
    PREPEND_TO_STRING(GCC_OPT "-march=${GOOD_ARCH} -mtune=${GOOD_ARCH}")
  ENDIF(GOOD_ARCH)
#   MESSAGE("GOOD_ARCH = ${GOOD_ARCH}")
ENDIF(USING_GCC OR USING_GPP AND NOT PASSED_FIRST_CONFIGURE)

# MESSAGE("CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
# MESSAGE("APPLE = ${APPLE}")
# MESSAGE("LINUX = ${LINUX}")

##############################################################
## Set the defaults
##############################################################

# MESSAGE("CMAKE_C_COMPILER   = ${CMAKE_C_COMPILER}")
# MESSAGE("CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")

# MESSAGE("USING_GCC  = ${USING_GCC}")
# MESSAGE("USING_GPP  = ${USING_GPP}")
# MESSAGE("USING_ICC  = ${USING_ICC}")
# MESSAGE("USING_ICPC = ${USING_ICPC}")
# MESSAGE("CMAKE version = ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}")
# MESSAGE("CMAKE_SYSTEM = ${CMAKE_SYSTEM}")
# MESSAGE("CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")

MACRO(ADD_COMPILER_FLAG COMPILER FLAGS NEW_FLAG)
  IF(${COMPILER})
    PREPEND_TO_STRING(${FLAGS} ${${NEW_FLAG}})
  ENDIF(${COMPILER})
ENDMACRO(ADD_COMPILER_FLAG)

ADD_COMPILER_FLAG(USING_ICC    C_FLAGS_RELEASE INTEL_OPT)
ADD_COMPILER_FLAG(USING_ICPC CXX_FLAGS_RELEASE INTEL_OPT)
ADD_COMPILER_FLAG(USING_GCC    C_FLAGS_RELEASE GCC_OPT)
ADD_COMPILER_FLAG(USING_GPP  CXX_FLAGS_RELEASE GCC_OPT)

IF(UNIX)
  APPEND_TO_STRING(C_FLAGS "-fPIC")
  APPEND_TO_STRING(CXX_FLAGS "-fPIC")
ENDIF(UNIX)

set_flags(CMAKE_C_FLAGS         C_FLAGS)
set_flags(CMAKE_C_FLAGS_DEBUG   C_FLAGS_DEBUG)
set_flags(CMAKE_C_FLAGS_RELEASE C_FLAGS_RELEASE)

set_flags(CMAKE_CXX_FLAGS         CXX_FLAGS)
set_flags(CMAKE_CXX_FLAGS_DEBUG   CXX_FLAGS_DEBUG)
set_flags(CMAKE_CXX_FLAGS_RELEASE CXX_FLAGS_RELEASE)



