#ifndef CUDA_KERNELS_FOR_RENDERING_HPP
#define CUDA_KERNELS_FOR_RENDERING_HPP

#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

#include "target.hpp"
#include "backgroundsf.hpp"
#include "tracerf.hpp"
#include "compoundf.hpp"
#include "initializers.hpp"
#include "vector3D.hpp"
#include "BVHf.hpp"

template <class Tptr>
__global__ void completeRender(sf::Uint8 *pixels,
        int width, int height, 
        int depth, int samples,
        Tptr** targetHolder,  // BVHTree** or TargetList**
        BackgroundColor** background,
        WindowVectors* window, 
        curandState* randState) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;

    int idx = width * j + i;

    Vector3D color = 255.0f/samples * TracePixelRnd(window, i, j, *targetHolder, depth, samples, *background, randState + idx);

    idx = idx << 2;
    
    pixels[idx] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    pixels[idx + 3] = 255;
}

/**
 * @brief Calculates pixel values and adds them to pixels
 * 
 * @param pixels Allocated Vector3D array of size width*height
 * @param width 
 * @param height 
 * @param depth Number of iterations for a single ray
 * @param samples Number of iterations for a single pixel
 * @param tree BVH structure for objects in the scene
 * @param background Defines background color for escaping rays
 * @param window Defines image ray positioning
 * @param randState Defines the random number generator for each thread
 */
__global__ void completeRender(Vector3D* pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState);

__global__ void renderPixel(Vector3D* color, 
    int x, int y, 
    int width, int height,
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState);

/**
 * @brief Calculates pixels sequentially starting from index "pixelIdx",
 *          gridDim.y in kernel launch sets the number of pixels
 * 
 * @param color Allocated Vector3D array of size width*height, unnormalized
 * @param pixelIdx Starting index: x + y*width
 * @param width 
 * @param height 
 * @param depth Number of iterations for a single ray
 * @param samples Number of iterations for a single pixel
 * @param tree BVH structure for objects in the scene
 * @param background Defines background color for escaping rays
 * @param window Defines image ray positioning
 * @param randState Defines the random number generator for each thread
 */
__global__ void renderPixels(Vector3D* color, 
    int pixelIdx,
    int width, int height,
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState);


__global__ void RealTimeRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState, double* darray);

__global__ void RealTimeUpdateRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState, double* darray, float frameIdx);

// ###############################################
// # INITIALIZATION & MEMORY RELEASE
// ###############################################

/**
 * @brief Auxiliary kernel for copying file contents to the rendered used in fileOperations.hpp
 * 
 * @param list 
 * @param shapes 
 * @param vertices 
 * @param fVertices 
 * @param fColors 
 * @param defaultColor
 */
__global__ void generateTriangles(TargetList** list, 
                                Vector3D* vertices, int* fVertices, 
                                Vector3D* fColors, size_t fCount, 
                                Vector3D* defaultColor);

__global__ void generateCompounds(Compound** list, Vector3D* vertices, 
                                    int* fVertices, Vector3D* fColors, 
                                    size_t fCount, Vector3D* defaultColor);

__global__ void addCompoundsToTargetlist(Compound** compounds, size_t compoundCount, TargetList** list);

__global__ void initializeBG(BackgroundColor** background, backgroundType type);

__global__ void initializeTargets(Target** targets, TargetList** list, int capacity, sceneType type);

__global__ void buildBVH(TargetList** listptr, BVHTree** tree);

__global__ void initializeRand(curandState* randState, int width, int height, int seed = 1889);
__global__ void initializeRandSamples(curandState* randState, int seed = 1889);

__global__ void releaseBG(BackgroundColor** background);

__global__ void releaseTargets(Target** targets, TargetList** list);

__global__ void releaseBVH(Target** targets, TargetList** list, BVHTree* tree);


#endif