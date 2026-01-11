/// \file       settings.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Provides the basic simulation and rendering settings for easy access.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once
#include <SDL3/SDL_log.h>

// Rendering settings
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define BLOCKDIM 16
#define MAX_FPS 60

// Other constants for settings
constexpr double ASPECT_RATIO = IMAGE_WIDTH / static_cast<double>(IMAGE_HEIGHT);
constexpr unsigned int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr float tickDelta = 1000.0f / static_cast<float>(MAX_FPS);

// Function macros
#define sdlprint(...) SDL_LogMessage(0, SDL_LOG_PRIORITY_DEBUG, __VA_ARGS__)

// CUDA macros
inline cudaError_t cuda_err = cudaSuccess; // Global variable to ensure these macros always work.
// Check a specific call for an error
#define CERR(fn) \
    cuda_err = fn; \
    if (cuda_err != cudaSuccess) SDL_LogMessage(1, SDL_LOG_PRIORITY_CRITICAL, "CUDA error: %s\n", cudaGetErrorString(cuda_err))
// Check the last error
#define CLERR() CERR(cudaGetLastError())