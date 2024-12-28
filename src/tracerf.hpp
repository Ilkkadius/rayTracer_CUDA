#ifndef TRACER_CUDA_HPP
#define TRACER_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vector3D.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "scatteringf.hpp"
#include "targetList.hpp"
#include "auxiliaryf.hpp"
#include "BVHf.hpp"


__device__ HitInfo closestHit(const Ray& ray, TargetList* listptr);

__device__ HitInfo closestHit(const Ray& ray, BVHTree* tree);

__device__ Vector3D Trace(const Ray& ray, TargetList** listptr, BackgroundColor* background, int depth, curandState randState);

__device__ Vector3D Trace(const Ray& ray, BVHTree* tree, BackgroundColor* background, int depth, curandState& randState);


__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, TargetList** listptr, 
                        int depth, int samples, BackgroundColor* background, curandState& randState);

__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, BVHTree* tree, 
                        int depth, int samples, BackgroundColor* background, curandState& randState);

__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, BVHTree* tree, 
                        int depth, BackgroundColor* background, curandState& randState);


#endif