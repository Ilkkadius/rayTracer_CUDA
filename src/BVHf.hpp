#ifndef BOUNDING_VOLUME_HIERARCHY_CUDA_HPP
#define BOUNDING_VOLUME_HIERARCHY_CUDA_HPP

#include <cuda_runtime.h>

#include "scatteringf.hpp"
#include "targetList.hpp"

constexpr float maxDist = 1e20f;

class Node{
public:
    Vector3D minBox, maxBox;
    uint first, targetCount;

    __device__ float getsHit(const Ray& ray) const;
};

class BVHTree{
public:

    uint nodeCount, targetCount;
    Node* nodes;
    Target** targets;
    uint* tIdx;

    __device__ BVHTree(TargetList** listptr);

    __device__ void findCollision(const Ray& ray, HitInfo& hit) const;

    __device__ void clear();

private:

    __device__ void setNodeBounds(uint nodeIdx);

    __device__ void subDivision();

    __device__ void simpleSplit(Node* node, int& axis, float& splitPosition);
};

#endif