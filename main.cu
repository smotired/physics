#include "viewport.cuh"
#include "programs/particles/particles.cuh"

int main(int argc, char *argv[]) {
    ParticleProgram program;
    Viewport viewport(program);

    // Just jump straight into the viewport loop for now
    return viewport.StartViewport();
}
