#include "viewport.cuh"
#include "programs/basic/basic.cuh"

int main(int argc, char *argv[]) {
    BasicProgram program;
    Viewport viewport(program);

    // Just jump straight into the viewport loop for now
    return viewport.StartViewport();
}
