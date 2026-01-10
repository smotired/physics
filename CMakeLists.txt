cmake_minimum_required(VERSION 4.0)
project(physics CUDA)

set(CMAKE_CUDA_STANDARD 20)

# SDL
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
add_subdirectory(vendored/SDL EXCLUDE_FROM_ALL)

# Main executable
add_executable(physics main.cu)

set_target_properties(physics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Link SDL
target_link_libraries(physics PRIVATE SDL3::SDL3)

