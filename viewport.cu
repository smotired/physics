/// \file       viewport.cu
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Implementations for the viewport class.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#include "viewport.cuh"
#include "settings.cuh"

int Viewport::ShowViewport() {
    // Initialize video subsystems
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", "Unable to initialize SDL!", nullptr);
        return 1;
    }

    // Create the window
    window = SDL_CreateWindow("Physics Demo", IMAGE_WIDTH, IMAGE_HEIGHT, 0);
    if (!window) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", "Unable to create window.", window);
        Cleanup();
        return 1;
    }

    // Create the renderer.
    renderer = SDL_CreateRenderer(window, nullptr); // second param is renderer device
    if (!renderer) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", "Unable to create renderer.", window);
        Cleanup();
        return 1;
    }

    // Start the program loop
    bool running = true;
    while (running) {
        // Poll for events
        SDL_Event event{ 0 };
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                // On quit event, stop running.
                case SDL_EVENT_QUIT:
                    running = false;
                    break;
                default:
                    break;
            }
        }

        // Draw to screen
        SDL_SetRenderDrawColor(renderer, 0, 255, 255, 255);
        SDL_RenderClear(renderer);

        // Swap buffers to present
        SDL_RenderPresent(renderer);
    }

    // Clean up
    Cleanup();
    return 0;
}

void Viewport::Cleanup() {
    SDL_DestroyRenderer(renderer);
    renderer = nullptr;
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Quit();
}
