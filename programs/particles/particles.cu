/// \file       particles.cu
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Implementations for particles.cuh
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#include "particles.cuh"

#include <curand_kernel.h>

#include "utils/float3.cuh"
#include "utils/rng.cuh"

__device__ bool bounded(int3 p) {
    return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < FIELD_WIDTH && p.y < FIELD_HEIGHT; // and z later
}

__global__ void Simulate(FieldData const& fieldData) {
    // Get target particle
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pX >= FIELD_WIDTH || pY >= FIELD_HEIGHT) return;
    const unsigned int pI = pY * FIELD_WIDTH + pX;

    // Don't deal with massless particles
    constexpr float epsMass = 1e-6f;
    if (fieldData.mass[pI] <= epsMass) return;

    // Extract particle information
    const float3 pos = float3(pX, pY, 0);
    const float mass = fieldData.mass[pI];
    const float3 momentum = fieldData.momentum[pI];
    float3 velocity = momentum * (1.0f / mass);

    // Calculate acceleration from gravity
    float3 grav_accel = F3_ZERO;
    for (unsigned int x = 0; x < FIELD_WIDTH; x++) {
        for (unsigned int y = 0; y < FIELD_HEIGHT; y++) {
            if (x == pX && y == pY) continue;

            const float3 offset = pos - float3(x, y, 0);
            const float3 direction = asNorm(offset);
            const float r2 = lengthsq(offset);

            // Standard gravity formula but with acceleration canceled.
            grav_accel += -direction * (fieldData.mass[y * FIELD_WIDTH + x] / r2 * GRAVITATIONAL_CONSTANT);
        }
    }
    velocity += grav_accel;

    // Decide exact position if particles at this field move this way
    const float3 next = pos + velocity;

    // Get the four closest points.
    const int3 nw = make_int3( floor(next.x), floor(next.y), 0 );
    const int3 ne = make_int3( nw.x + 1, nw.y, 0 );
    const int3 sw = make_int3( nw.x, nw.y + 1, 0 );
    const int3 se = make_int3( nw.x + 1, nw.y + 1, 0 );

    // Lerp
    const float lx = next.x - nw.x;
    const float ly = next.y - nw.y;

    // Split mass based on area. These functions assume the size of each pixel is 1 unit.
    const float nwArea = (1-lx) * (1-ly);
    const float neArea = (1-lx) *    ly;
    const float swArea =    lx  * (1-ly);
    const float seArea =    lx  *    ly;

    // Apply to the next buffers with conservation of momentum
    if (bounded(nw)) {
        const unsigned int nwI = nw.y * FIELD_WIDTH + nw.x;
        atomicAdd(&fieldData.nextMass[nwI], mass * nwArea);
        atomicF3Add(fieldData.nextMomentum[nwI], momentum * nwArea); // a particle of 0.25x mass but same velocity would have 0.25x momentum
    }
    if (bounded(ne)) {
        const unsigned int neI = ne.y * FIELD_WIDTH + ne.x;
        atomicAdd(&fieldData.nextMass[neI], mass * neArea);
        atomicF3Add(fieldData.nextMomentum[neI], momentum * neArea);
    }
    if (bounded(sw)) {
        const unsigned int swI = sw.y * FIELD_WIDTH + sw.x;
        atomicAdd(&fieldData.nextMass[swI], mass * swArea);
        atomicF3Add(fieldData.nextMomentum[swI], momentum * swArea);
    }
    if (bounded(se)) {
        const unsigned int seI = se.y * FIELD_WIDTH + se.x;
        atomicAdd(&fieldData.nextMass[seI], mass * seArea);
        atomicF3Add(fieldData.nextMomentum[seI],  momentum * seArea);
    }
}

__global__ void PushFields(FieldData const& fieldData, const bool randomize) {
    // Get target particle
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pX >= FIELD_WIDTH || pY >= FIELD_HEIGHT) return;
    const unsigned int pI = pY * FIELD_WIDTH + pX;

    // If randomize is true, do that
    if (randomize) {
        // Initialize rng
        curandStateXORWOW_t rng;
        curand_init(pI, 0, 0, &rng);

        // Randomize mass and velocity
        const float mass = RandomFloat(&rng, MIN_INIT_MASS, MAX_INIT_MASS);
        const float momentumX = RandomFloat(&rng, -MAX_INIT_VEL, MAX_INIT_VEL) * mass;
        const float momentumY = RandomFloat(&rng, -MAX_INIT_VEL, MAX_INIT_VEL) * mass;

        // Save
        fieldData.mass[pI] = mass;
        fieldData.momentum[pI] = make_float3(momentumX, momentumY, 0); // temporarily have no Z velocity
    }

    // Otherwise pull from next array
    else {
        fieldData.mass[pI] = fieldData.nextMass[pI];
        fieldData.momentum[pI] = fieldData.nextMomentum[pI];
    }

    // Zero out next values
    fieldData.nextMass[pI] = 0;
    fieldData.nextMomentum[pI] = F3_ZERO;
}

__global__ void ConvertToImage(float const* mass, unsigned char* pixels) {
    // For now: Assume that image size and field size are the same

    // Get target particle
    const unsigned int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pX >= FIELD_WIDTH || pY >= FIELD_HEIGHT) return;
    const unsigned int pI = pY * FIELD_WIDTH + pX;

    // Mass should fade from black to white
    const unsigned char color = min(static_cast<int>(mass[pI] * massThreshold * 256), 255);
    pixels[3 * pI + 0] = color;
    pixels[3 * pI + 1] = color;
    pixels[3 * pI + 2] = color;
}

void ParticleProgram::StepFrame(void *target) {
    CLERR();

    // Simulate
    Simulate<<<simBlocks, simThreads>>>(fieldData);
    CLERR();
    CERR(cudaDeviceSynchronize());

    // Copy buffer back
    PushFields<<<simBlocks, simThreads>>>(fieldData);
    CLERR();
    CERR(cudaDeviceSynchronize());

    // Set up thread count for images
    constexpr dim3 imgBlocks((IMAGE_WIDTH + BLOCKDIM - 1) / BLOCKDIM, (IMAGE_HEIGHT + BLOCKDIM - 1) / BLOCKDIM);
    constexpr dim3 imgThreads(BLOCKDIM, BLOCKDIM);

    // Allocate memory for the image
    unsigned char *image;
    CERR(cudaMalloc(&image, sizeof(unsigned char) * 3 * IMAGE_SIZE));

    // Launch the kernel
    ConvertToImage<<<imgBlocks, imgThreads>>>(fieldData.mass, image);
    CLERR();
    CERR(cudaDeviceSynchronize());

    // Write to texture
    CERR(cudaMemcpy(target, image, sizeof(unsigned char) * 3 * IMAGE_SIZE, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(image);
}