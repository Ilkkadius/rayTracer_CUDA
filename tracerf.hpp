#ifndef TRACER_CUDA_HPP
#define TRACER_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vector3D.hpp"
#include "rayf.hpp"
#include "backgroundsf.hpp"
#include "scatteringf.hpp"
#include "targetf.hpp"
#include "auxiliaryf.hpp"

__device__ HitInfo closestHit(const Ray& ray, targetList** listptr) {
    HitInfo info;
    targetList list = **listptr;
    for(int i = 0; i < list.size; i++) {
        Target target = list[i];
        float u = target.collision(ray);
        if(u > epsilon) {
            if(info.t > u || info.t < 0.0f) {
                info = HitInfo(ray, ray.at(u), target.normal(ray.at(u)), u);
            }
        }
    }
    return info;
}

__device__ Vector3D Trace(const Ray& ray, targetList** listptr, BackgroundColor* background, int depth, curandState randState) {
    Ray current = ray;
    Vector3D color(1.0f, 1.0f, 1.0f);
    HitInfo info;

    for(int i = 0; i < depth; i++) {
        info = closestHit(current, listptr);

        if(info.t > epsilon) {
            color = 0.7f * color;
            Vector3D p = info.point, n = info.normal;
            Vector3D dir = n + aux::randUnitVec(&randState);
            while(dir.lengthSquared() < 0.001f) {
                dir = n + aux::randUnitVec(&randState);
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