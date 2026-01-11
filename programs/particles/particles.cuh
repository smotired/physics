/// \file       particles.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Basic particle simulation.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once
#include "program.cuh"
#include "settings.cuh"

// Simulation settings
#define FIELD_WIDTH 1024
#define FIELD_HEIGHT 1024
#define MIN_INIT_MASS 0.0f
#define MAX_INIT_MASS 1.0f
#define MAX_INIT_VEL 0.0f

// Technical
#define FIELD_BLOCKDIM 16
constexpr unsigned int FIELD_SIZE = FIELD_WIDTH * FIELD_HEIGHT;
constexpr dim3 simBlocks((FIELD_WIDTH + FIELD_BLOCKDIM - 1) / FIELD_BLOCKDIM, (FIELD_HEIGHT + FIELD_BLOCKDIM - 1) / FIELD_BLOCKDIM);
constexpr dim3 simThreads(FIELD_BLOCKDIM, FIELD_BLOCKDIM);
constexpr float massThreshold = 1.0f / static_cast<float>(MAX_INIT_MASS) * 0.5f; // multiplier for mass, where a value of 1 as a result would be white. so if max init mass is 1, 2 is white.

/// Information about the fields. Tracks pointers
struct FieldData {
    /// Total mass over each region
    float *mass;

    /// Total momentum of particles in each region
    float3 *momentum;

    /// Total mass over each region in the next frame
    float *nextMass;

    /// Total momentum of particles in next region in the next frame.
    float3 *nextMomentum;
};

/// Kernel function that performs one frame of simulation.
/// @param fieldData Information about the fields we care about. Passed by value.
__global__ void Simulate(FieldData const& fieldData);

/// Kernel to push the fields from the next buffer to the current buffer, and zero-out the next buffer.
/// @param fieldData Information about the fields we care about. Passed by value.
/// @param randomize If true, randomize the current buffer instead of passing it along.
__global__ void PushFields(FieldData const& fieldData, bool randomize = false);

/// Kernel to convert mass into final pixel data.
__global__ void ConvertToImage(float const* mass, unsigned char* pixels);

/// Simulates particles in a vacuum with tensor fields.
class ParticleProgram : public Program {
    FieldData fieldData;
public:
    ParticleProgram() : Program() {
        // Allocate mass and velocity arrays on device
        CERR(cudaMalloc(&fieldData.mass, FIELD_SIZE * sizeof(float)));
        CERR(cudaMalloc(&fieldData.momentum, FIELD_SIZE * sizeof(float3)));
        CERR(cudaMalloc(&fieldData.nextMass, FIELD_SIZE * sizeof(float)));
        CERR(cudaMalloc(&fieldData.nextMomentum, FIELD_SIZE * sizeof(float3)));

        // Start a kernel to randomize
        PushFields<<<simBlocks, simThreads>>>(fieldData, true);
        CLERR();
        CERR(cudaDeviceSynchronize());
    }

    ~ParticleProgram() override {
        // Free all the device memory
        CERR(cudaFree(fieldData.mass));
        CERR(cudaFree(fieldData.momentum));
        CERR(cudaFree(fieldData.nextMass));
        CERR(cudaFree(fieldData.nextMomentum));
    }

    /// Render a single frame of the program.
    /// @param target Array of target pixels to write to, as 3WH 8-bit chars.
    void StepFrame(void* target) override;
};