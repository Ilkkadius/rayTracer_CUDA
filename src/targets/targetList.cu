#include "targetList.hpp"

__device__ TargetList::TargetList(Target** targets_, int N, int maxN) {
    targets = targets_; size = N; capacity = maxN;
}

__device__ TargetList::TargetList(int capacity_) : targets(NULL), size(0), capacity(capacity_) {}

__device__ Target* TargetList::operator[](int i) {
    return *(targets + i); // NOT *(*targets + i) ?
}

__device__ void TargetList::append(Target** additional, size_t amount) {
    for(size_t i = 0; i < amount; i++) {
        if(size < capacity) {
            targets[size] = additional[i];
            size++;
        } else {
            delete additional[i]->shape;
            delete additional[i];
        }
    }
    delete[] additional;
}

__device__ void TargetList::findCollision(const Ray& ray, HitInfo& hit) const {
    for(int i = 0; i < size; i++) {
        Target* target = targets[i];
        float u = target->collision(ray);
        if(u > epsilon) {
            if(hit.t > u || hit.t < 0.0f) {
                hit = HitInfo(ray.dir, ray.at(u), target->normal(ray.at(u)), u, target);
            }
        }
    }
}

