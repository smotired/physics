/// \file       program.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief The program class represents a single kind of program.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once

/// Abstract base class for a Program.
class Program {
public:
    virtual ~Program() = default;

    /// Render a single frame of the program.
    /// @param target Array of target pixels to write to, as 3WH 8-bit chars.
    virtual void StepFrame(void* target) = 0;
};