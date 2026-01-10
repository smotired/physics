/// \file       viewport.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief The Viewport class provides functionality for displaying program
/// information to the screen.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once

#include "program.cuh"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

/// A window for displaying program information
class Viewport {
    /// Program this viewport is running
    Program &program;

    /// Main window
    SDL_Window *window = nullptr;

    /// Main renderer
    SDL_Renderer *renderer = nullptr;

    /// The texture for the screen
    SDL_Texture *screen = nullptr;

    /// Initialize the program.
    int Initialize();

    /// Clean up window and renderer, and stop program.
    void Cleanup();

    /// Draw a single frame
    void DrawLoop() const;
public:
    explicit Viewport(Program &program) : program(program) {};

    /// Starts the display program and runs until it finishes.
    int StartViewport();
};