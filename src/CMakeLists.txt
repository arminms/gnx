# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Armin Sobhani
#

## Pass the version as a preprocessor
#
add_definitions(-DGNX_VERSION="${PROJECT_VERSION}")

find_package(
  CLI11 CONFIG
  HINTS $ENV{HOME} $ENV{HOME}/.local /usr/local /opt/local /opt
)
if(NOT CLI11_FOUND)
    message(STATUS "Fetching CLI11 library...")
    FetchContent_Declare(
      cli11
      GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
      GIT_TAG        v2.6.2
    )
    FetchContent_MakeAvailable(cli11)
endif()

## defining target for gnx executable
#
add_executable(gnx_cli
  ansi.hpp
  bgzip.hpp
  bgzip.cpp
  complement.hpp
  complement.cpp
  gnx.hpp
  gnx.cpp
  prefix.hpp
  prefix.cpp
)
set_target_properties(gnx_cli PROPERTIES OUTPUT_NAME gnx)

if(CUDAToolkit_FOUND)
  set_source_files_properties(
    gnx.cpp
    complement.cpp
    PROPERTIES LANGUAGE CUDA
  )
  target_compile_options(gnx_cli PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
  )
  # ## Use fatbin to support all architectures
  # #
  # set_property(TARGET gnx_cli PROPERTY CUDA_ARCHITECTURES all)
endif()

if(hip_FOUND)
  set_source_files_properties(
    gnx.cpp
    complement.cpp
    PROPERTIES LANGUAGE HIP
  )
  target_compile_options(gnx_cli PRIVATE
    -Wno-unused-result -Wno-pass-failed
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx942>
  )
endif()

## defining link libraries for gnx_cli
#
target_link_libraries(gnx_cli PRIVATE
  ${PROJECT_NAME}::${PROJECT_NAME}
  CLI11::CLI11
)

install(TARGETS gnx_cli
  # EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
