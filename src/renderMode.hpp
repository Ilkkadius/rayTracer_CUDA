#ifndef RENDERMODE_CUDA_HPP
#define RENDERMODE_CUDA_HPP

#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "auxiliaryf.hpp"
#include "kernelSet.hpp"

enum class RenderMode {
    Single_full,
    Partial_full,
    Partial_pixel
};

namespace Mode {
void FullRender(int width, int height, int tx, int ty, timepoint& start, curandState** randState_ptr, Vector3D* results, 
                int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow);
void partialFullRender(int width, int height, int tx, int ty, timepoint& start, curandState** randState_ptr, Vector3D* results, 
                    int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow);
void partialPixelRender(int width, int height, int tx, int ty, timepoint& start, curandState** randState_ptr, Vector3D* results, 
                    int depth, int samples, BVHTree** tree, BackgroundColor** background_d, WindowVectors* cudaWindow);
};
#endif