/// \file       settings.cuh
/// \author     Sam Hill (whatinthesamhill.dev)
/// \version    1.0
/// \date       January 10, 2026
///
/// \brief Provides the basic simulation and rendering settings for easy access.
///
/// Copyright (c) 2026 Sam Hill. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or
/// sublicensing of this code or its derivatives is strictly prohibited.
#pragma once

// Program settings

// Rendering settings
#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

// Other constants for settings
constexpr double ASPECT_RATIO = IMAGE_WIDTH / static_cast<double>(IMAGE_HEIGHT);