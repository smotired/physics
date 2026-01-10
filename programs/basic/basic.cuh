/// \file       basic.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Basic program for drawing just a test gradient.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once
#include "program.cuh"

/// Basic image that pulses between two test gradients.
class BasicProgram : public Program {
    int blue = 0;
    int blueDirection = 1;
public:
    /// Render a single frame of the program.
    /// @param target Array of target pixels to write to, as 3WH 8-bit chars.
    void StepFrame(void* target) override;
};

/// Render a very basic testing image that trends toward red on the X axis and green on the Y axis.
/// @param blue: Blue color integer from 0 to 255
/// @param output: Array of size 3xy of chars for output pixels.
__global__ void DefaultImage(char blue, char *output);