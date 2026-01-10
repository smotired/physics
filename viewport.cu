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

#include "kernels.cuh"
#include "settings.cuh"

int Viewport::Initialize() {
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

    // Create the renderer
    renderer = SDL_CreateRenderer(window, nullptr); // second param is renderer device
    if (!renderer) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", "Unable to create renderer.", window);
        Cleanup();
        return 1;
    }

    // Create the screen texture
    screen = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, IMAGE_WIDTH, IMAGE_HEIGHT);

    SDL_SetLogPriorities(SDL_LOG_PRIORITY_DEBUG);

    return 0;
}

void Viewport::DrawLoop() const {
    CLERR();

    // Allocate memory for the image
    char *image;
    CERR(cudaMalloc(&image, sizeof(char) * 3 * IMAGE_SIZE));

    // Set up thread count
    constexpr dim3 numBlocks((IMAGE_WIDTH + BLOCKDIM - 1) / BLOCKDIM, IMAGE_HEIGHT + BLOCKDIM - 1 / BLOCKDIM);
    constexpr dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);

    // Launch the kernel
    DefaultImage<<<numBlocks, threadsPerBlock>>>(image);
    CLERR();
    CERR(cudaDeviceSynchronize());

    // Prepare texture for writes
    void *target; // Pointer to pixels array
    int pitch; // Length of a row of pixels. Texture size = image size so this is unused.
    SDL_LockTexture(screen, nullptr, &target, &pitch);

    // Write to texture
    CERR(cudaMemcpy(target, image, sizeof(char) * 3 * IMAGE_SIZE, cudaMemcpyDeviceToHost));
    SDL_UnlockTexture(screen);

    // Draw to screen
    SDL_RenderTexture(renderer, screen, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    // Cleanup
    cudaFree(image);
}


void Viewport::Cleanup() {
    SDL_DestroyTexture(screen);
    screen = nullptr;
    SDL_DestroyRenderer(renderer);
    renderer = nullptr;
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Quit();
}

int Viewport::StartViewport() {
    if (const int initialize_error = Initialize(); initialize_error)
        return initialize_error;

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
        DrawLoop();
    }

    // Clean up
    Cleanup();
    return 0;
}