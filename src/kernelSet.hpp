#ifndef CUDA_KERNELS_FOR_RENDERING_HPP
#define CUDA_KERNELS_FOR_RENDERING_HPP

#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

#include "targetf.hpp"
#include "backgroundsf.hpp"
#include "tracerf.hpp"
#include "compoundf.hpp"
#include "initializers.hpp"
#include "vector3D.hpp"
#include "BVHf.hpp"

__global__ void completeRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    TargetList** list,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState);

__global__ void completeRender(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    BVHTree** tree,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState);

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


__global__ void renderHalf(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    TargetList** list,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState, bool left);


__global__ void renderQuarter(sf::Uint8 *pixels, 
    int width, int height, 
    int depth, int samples,
    TargetList** list,
    BackgroundColor** background, 
    WindowVectors* window, 
    curandState* randState, int quarter);

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
__global__ void generateTargets(TargetList** list, Shape** shapes, 
                                Vector3D* vertices, int* fVertices, 
                                Vector3D* fColors, size_t fCount, 
                                Vector3D* defaultColor);

__global__ void generateCompounds(Compound** list, Vector3D* vertices, 
                                    int* fVertices, Vector3D* fColors, 
                                    size_t fCount, Vector3D* defaultColor);

__global__ void addCompoundsToTargetlist(Compound** compounds, size_t compoundCount, TargetList** list, Shape** shapes);

__global__ void initializeBG(BackgroundColor** background);

__global__ void initializeTargets(Target** targets, TargetList** list, Shape** shapes, int capacity);

__global__ void buildBVH(TargetList** listptr, BVHTree** tree);

__global__ void initializeRand(curandState* randState, int width, int height, int seed = 1889);
__global__ void initializeRandSamples(curandState* randState, int seed = 1889);

__global__ void releaseBG(BackgroundColor** background);

__global__ void releaseTargets(Target** targets, TargetList** list, Shape** shapes);

__global__ void releaseBVH(Target** targets, TargetList** list, Shape** shapes, BVHTree* tree);


#endif