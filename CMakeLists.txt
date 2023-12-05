cmake_minimum_required (VERSION 3.8)

project ("MoellerTrumboreIntersectionAlgorithm" LANGUAGES CXX CUDA)


file(GLOB_RECURSE SOURCE_FILES
    "src/*.cu"
    "src/*.cpp"
    "include/*.h"
)

add_executable (MoellerTrumboreIntersectionAlgorithm ${SOURCE_FILES})


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MoellerTrumboreIntersectionAlgorithm PROPERTY CXX_STANDARD 20)
endif()
