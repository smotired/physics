/// \file       kernels.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Top-level kernel functions.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once

#include "settings.cuh"

/// Render a very basic testing image that trends toward red on the X axis and green on the Y axis.
/// \param output: Array of size 3xy of floats for output pixels.
__global__ void DefaultImage(float *output) {
    // Get target pixel
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pX >= IMAGE_WIDTH || pY >= IMAGE_HEIGHT) return;
    const unsigned int pI = pY * IMAGE_WIDTH + pX;

    constexpr float OneOverWidthMinusOne = 1 / static_cast<float>(IMAGE_WIDTH);
    constexpr float OneOverHeightMinusOne = 1 / static_cast<float>(IMAGE_HEIGHT);

    // Approach red on the X axis and green on the Y axis
    output[3 * pI + 0] = pX * OneOverWidthMinusOne;
    output[3 * pI + 1] = pY * OneOverHeightMinusOne;
    output[3 * pI + 2] = 0;
}