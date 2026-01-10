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

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

class Viewport {
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;

    /// Clean up window and renderer, and stop program.
    void Cleanup();
public:
    /// Starts the display program.
    int ShowViewport();
};