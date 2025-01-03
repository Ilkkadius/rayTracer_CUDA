#include "tracerf.hpp"


__device__ HitInfo closestHit(const Ray& ray, TargetList* listptr) {
    HitInfo hit;
    listptr->findCollision(ray, hit);
    return hit;
}

__device__ HitInfo closestHit(const Ray& ray, BVHTree* tree) {
    HitInfo hit;
    tree->findCollision(ray, hit);
    return hit;
}

__device__ Vector3D Trace(const Ray& ray, TargetList** listptr, BackgroundColor* background, int depth, curandState randState) {
    Ray current = ray;
    Vector3D rayColor(1.0f, 1.0f, 1.0f);
    HitInfo info;

    for(int i = 0; i < depth; i++) {
        (*listptr)->findCollision(current, info);

        if(info.t > epsilon) {
            if(info.target->isRadiant()) {
                return rayColor * info.target->emission();
            } else {
                rayColor = info.target->color * rayColor;
                Vector3D p = info.point, n = info.normal;
                Vector3D dir = n + aux::randUnitVec(&randState);
                while(dir.lengthSquared() < 0.001f) {
                    dir = n + aux::randUnitVec(&randState);
                }
                current = Ray(dir, p);
            }
        } else {
            return rayColor * background->colorize(current);
        }
    }
    
    return Vector3D(0.0f, 0.0f, 0.0f);
}

__device__ Vector3D Trace(const Ray& ray, BVHTree* tree, BackgroundColor* background, int depth, curandState& randState) {
    Ray current = ray;
    Vector3D rayColor(1.0f, 1.0f, 1.0f);
    HitInfo info;

    for(int i = 0; i < depth; i++) {
        info = closestHit(current, tree);

        if(info.t > epsilon) {
            if(info.target->isRadiant()) {
                return rayColor * info.target->emission();
            } else {
                rayColor = info.target->color * rayColor;
                Vector3D p = info.point, n = info.normal;
                Vector3D dir = n + aux::randUnitVec(&randState);
                while(dir.lengthSquared() < 0.001f) {
                    dir = n + aux::randUnitVec(&randState);
                }
                current = Ray(dir, p);
            }
        } else {
            return rayColor * background->colorize(current);
        }
    }
    
    return Vector3D(0.0f, 0.0f, 0.0f);
}


__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, TargetList** listptr, 
                        int depth, int samples, BackgroundColor* background, curandState& randState) {
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
    color = color/float(samples);
    if(color.max() > 1) {
        color = color/color.max();
    }
    return color;
}

__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, BVHTree* tree, 
                        int depth, int samples, BackgroundColor* background, curandState& randState) {
    Vector3D color;
    int k = 0;
    Vector3D start = window->starter_, xdiff = window->xVec_, ydiff = window->yVec_, eye = window->eye_;
    while(k < samples) {
        Ray rndRay = Ray(start
                + (float(x) - aux::randUnitFloat(&randState)) * xdiff 
                + (float(y) - aux::randUnitFloat(&randState)) * ydiff, 
                eye);
        color += Trace(rndRay, tree, background, depth, randState);
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
                        int depth, BackgroundColor* background, curandState& randState) {
    Vector3D start = window->starter_, xdiff = window->xVec_, ydiff = window->yVec_, eye = window->eye_;
    Ray rndRay = Ray(start
            + (float(x) - aux::randUnitFloat(&randState)) * xdiff 
            + (float(y) - aux::randUnitFloat(&randState)) * ydiff, 
            eye);
    return Trace(rndRay, tree, background, depth, randState);
}
