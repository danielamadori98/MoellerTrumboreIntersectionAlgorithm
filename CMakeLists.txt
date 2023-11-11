﻿cmake_minimum_required (VERSION 3.8)

# Abilita Ricaricamento rapido per i compilatori MSVC, se supportato.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

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