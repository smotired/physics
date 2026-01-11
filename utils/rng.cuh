/// \file       rng.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Defines a device-accessible function to get a pseudo-random float.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once
#include <curand_kernel.h>

/// Return a random float in the range [0, 1) using the cuRAND library.
/// @param rng CuRand XORWOW state initialized for this pixel.
/// @param min Start range, inclusive. Default 0.
/// @param max Stop range, exclusive. Default 1.
__device__ inline float RandomFloat(curandStateXORWOW_t *rng, const float min = 0.0f, const float max = 1.0f) {
    constexpr float rmax = 0x1.fffffep-1;
    const float r = static_cast<float>(curand(rng)) * 0x1p-32f;
    const float v = r < rmax ? r : rmax;
    return min + v * (max - min);
}