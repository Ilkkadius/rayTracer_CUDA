#include "targetList.hpp"

__device__ TargetList::TargetList(Target** targets_, int capacity_) : targets(targets_), size(0), capacity(capacity_) {}

__device__ void TargetList::append(Target** additional, size_t amount) {
    for(size_t i = 0; i < amount; i++) {
        if(size < capacity) {
            targets[size] = additional[i];
            size++;
        } else {
            delete additional[i];
        }
    }
    delete[] additional;
}

__device__ void TargetList::findCollision(const Ray& ray, HitInfo& hit) const {
    HitInfo tempHit;
    for(int i = 0; i < size; i++) {
        if(targets[i]->rayCollision(ray, &tempHit) && (hit.t < 0.0f || (tempHit.t > epsilon && tempHit.t < hit.t))) {
            hit = tempHit;
        }
    }
}

