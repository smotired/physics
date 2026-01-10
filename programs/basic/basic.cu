/// \file       basic.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Implementations for basic.cuh
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#include "basic.cuh"
#include "settings.cuh"

void BasicProgram::StepFrame(void *target) {
    CLERR();

    // Allocate memory for the image
    char *image;
    CERR(cudaMalloc(&image, sizeof(char) * 3 * IMAGE_SIZE));

    // Set up thread count
    constexpr dim3 numBlocks((IMAGE_WIDTH + BLOCKDIM - 1) / BLOCKDIM, IMAGE_HEIGHT + BLOCKDIM - 1 / BLOCKDIM);
    constexpr dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);

    // Launch the kernel
    DefaultImage<<<numBlocks, threadsPerBlock>>>(static_cast<char>(blue), image);
    CLERR();
    CERR(cudaDeviceSynchronize());

    // Write to texture
    CERR(cudaMemcpy(target, image, sizeof(char) * 3 * IMAGE_SIZE, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(image);

    // Update blue
    if (blueDirection > 0) {
        if (++blue >= 255)
            blueDirection = -1;
    }
    else if (blueDirection < 0) {
        if (--blue <= 0)
            blueDirection = 1;
    }
}

__global__ void DefaultImage(const char blue, char *output) {
    // Get target pixel
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pX >= IMAGE_WIDTH || pY >= IMAGE_HEIGHT) return;
    const unsigned int pI = pY * IMAGE_WIDTH + pX;

    constexpr float OneOverWidth = 1 / static_cast<float>(IMAGE_WIDTH);
    constexpr float OneOverHeight = 1 / static_cast<float>(IMAGE_HEIGHT);

    // Approach red on the X axis and green on the Y axis
    output[3 * pI + 0] = 256 * pX * OneOverWidth;
    output[3 * pI + 1] = 256 * pY * OneOverHeight;
    output[3 * pI + 2] = blue;
}