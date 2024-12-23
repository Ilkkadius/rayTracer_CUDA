#ifndef TRACER_CUDA_HPP
#define TRACER_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "linearAlgebraf.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "scatteringf.hpp"
#include "targetf.hpp"
#include "auxiliaryf.hpp"

__device__ HitInfo closestHit(Ray& ray, targetList** listptr) {
    HitInfo info;
    targetList tlist = **listptr;
    for(int i = 0; i < tlist.size; i++) {
        Target target = tlist[i];
        float u = target.collision(ray);
        if(u > epsilon) {
            if(info.t > u || info.t < 0.0f) {
                info = HitInfo(ray, ray.at(u), target.normal(ray.at(u)), u);
            }
        }
    }
    return info;
}

__device__ Vector3D Trace(Ray& ray, targetList** listptr, BackgroundColor* background, int depth, curandState randState) {
    Ray current = ray;
    Vector3D color(1.0f, 1.0f, 1.0f);

    for(int i = 0; i < depth; i++) {
        HitInfo info = closestHit(current, listptr);

        if(info.t > epsilon) {
            color = 0.7f * color;
            Vector3D p = info.point, n = info.normal;
            Vector3D dir = n + aux::randHemisphereVec(&randState, n);
            while(dir.lengthSquared() < 0.001f) {
                dir = n + aux::randHemisphereVec(&randState, n);
            }
            current = Ray(dir, p);
        } else {
            return color * background->colorize(current);
        }
    }
    
    return Vector3D(0.0f, 0.0f, 0.0f);
}

__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, targetList** listptr, 
                        int depth, int samples, BackgroundColor* background, curandState randState) {
    Vector3D color;
    int k = 0;
    Vector3D start = window->starter_, xdiff = window->xVec_, ydiff = window->yVec_, eye = window->eye_;
    while(k < samples) {
        Ray rndRay = Ray(start
                + (float(x) - aux::randUnitFloat(&randState)) * xdiff 
                + (float(y) - aux::randUnitFloat(&randState)) * ydiff, 
                eye);
        color += Trace(rndRay, listptr, background, depth, randState);
        k++;
    }
    return color/float(samples);
}

#endif