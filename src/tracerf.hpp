#ifndef TRACER_CUDA_HPP
#define TRACER_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vector3D.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "hitInfof.hpp"
#include "targetList.hpp"
#include "auxiliaryf.hpp"
#include "BVHf.hpp"


__device__ HitInfo closestHit(const Ray& ray, TargetList* listptr);

__device__ HitInfo closestHit(const Ray& ray, BVHTree* tree);

template <class Tptr>
__device__ Vector3D Trace(const Ray& ray, Tptr* targetHolder, BackgroundColor* background, int depth, curandState* randState) {
    Ray current = ray;
    Vector3D rayColor(1.0f, 1.0f, 1.0f);
    HitInfo info;

    for(int i = 0; i < depth; i++) {
        info = closestHit(current, targetHolder);

        if(info.t > epsilon) { // TODO: cases where 0 < t < epsilon
            if(info.emission > 0.0f) {
                return rayColor * info.emission;
            } else {
                rayColor = info.color * rayColor;
                Vector3D p = current.at(info.t), n = info.normal;
                Vector3D dir = n + aux::randUnitVec(randState);
                while(dir.lengthSquared() < 0.001f) {
                    dir = n + aux::randUnitVec(randState);
                }
                current = Ray(dir, p);
            }
        } else {
            return rayColor * background->colorize(current);
        }
    }
    
    return Vector3D(0.0f, 0.0f, 0.0f);
}

template <class Tptr>
__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, Tptr* targetHolder, 
                        int depth, int samples, BackgroundColor* background, curandState* randState) {
    Vector3D color(0.0f,0.0f,0.0f);
    int k = 0;
    Vector3D start = window->starter_, xdiff = window->xVec_, ydiff = window->yVec_, eye = window->eye_;
    while(k < samples) {
        Ray rndRay = Ray(start
                + (float(x) - aux::randUnitFloat(randState)) * xdiff 
                + (float(y) - aux::randUnitFloat(randState)) * ydiff, 
                eye);
        color += Trace(rndRay, targetHolder, background, depth, randState);
        k++;
    }
    /*
    color = color/float(samples);
    if(color.max() > 1) {
        color = color/color.max();
    }
    */
    return color;
}

__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, BVHTree* tree, 
                        int depth, BackgroundColor* background, curandState* randState);


#endif