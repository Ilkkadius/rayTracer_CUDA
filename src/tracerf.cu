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

__device__ Vector3D TracePixelRnd(WindowVectors* window, int x, int y, BVHTree* tree, 
                        int depth, BackgroundColor* background, curandState* randState) {
    Vector3D start = window->starter_, xdiff = window->xVec_, ydiff = window->yVec_, eye = window->eye_;
    Ray rndRay = Ray(start
            + (float(x) - aux::randUnitFloat(randState)) * xdiff 
            + (float(y) - aux::randUnitFloat(randState)) * ydiff, 
            eye);
    return Trace(rndRay, tree, background, depth, randState);
}
