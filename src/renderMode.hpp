#ifndef RENDERMODE_CUDA_HPP
#define RENDERMODE_CUDA_HPP

enum class RenderMode {
    Single_full,
    Partial_full,
    Partial_pixel
};

namespace Mode {
void FullRender(int width, int height, int tx, int ty, timepoint& start, curandState* randState_d, Vector3D* results, 
                int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow);
void partialFullRender(int width, int height, int tx, int ty, timepoint& start, curandState* randState_d, Vector3D* results, 
                    int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow);
void partialPixelRender(int width, int height, int tx, int ty, timepoint& start, curandState* randState_d, Vector3D* results, 
                    int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow);
};
#endif