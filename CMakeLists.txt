cmake_minimum_required(VERSION 4.0)
project(physics CUDA)

set(CMAKE_CUDA_STANDARD 20)

# --------------------------------
# SDL
# --------------------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
add_subdirectory(vendored/SDL EXCLUDE_FROM_ALL)

# --------------------------------
# Common modules
# --------------------------------

# --------------------------------
# Program modules
# --------------------------------

set(BASIC_PROGRAM
        programs/basic/basic.cu
        programs/basic/basic.cuh)

set(PARTICLES_PROGRAM
        programs/particles/particles.cu
        programs/particles/particles.cuh)

set(PROGRAMS ${BASIC_PROGRAM} ${PARTICLES_PROGRAM})

# --------------------------------
# Final executable
# --------------------------------

# Set up
add_executable(${PROJECT_NAME} main.cu viewport.cu ${PROGRAMS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR} utils)

# --------------------------------
# Link external libraries
# --------------------------------

# SDL
target_link_libraries(physics PRIVATE SDL3::SDL3)

# --------------------------------
# Final setup
# --------------------------------

# Ensure device functions are callable from anywhere
set_target_properties(physics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
