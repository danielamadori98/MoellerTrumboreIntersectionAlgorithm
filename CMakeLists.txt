cmake_minimum_required (VERSION 3.8)

set(PROJECT "MoellerTrumboreIntersectionAlgorithm")

project (${PROJECT} LANGUAGES CXX CUDA)


file(GLOB_RECURSE SOURCE_FILES
    "src/*.cu"
    "src/*.cpp"
    "include/*.hpp"
    "include/*.cuh"
    "include/lib/*.hpp"
    "include/lib/*.cuh"
)

add_executable (MoellerTrumboreIntersectionAlgorithm ${SOURCE_FILES})

set(CUDA_ARCHITECTURES 75)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MoellerTrumboreIntersectionAlgorithm PROPERTY CXX_STANDARD 20 CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
endif()
