#ifndef BOUNDING_VOLUME_HIERARCHY_CUDA_HPP
#define BOUNDING_VOLUME_HIERARCHY_CUDA_HPP

#include <cuda_runtime.h>

#include "scatteringf.hpp"
#include "targetList.hpp"

typedef unsigned int uint;

constexpr float maxDist = 1e20f;

class Node{
public:
    Vector3D minBox, maxBox;
    uint first, targetCount;

    __device__ float getsHit(const Ray& ray) const {
        Vector3D tmins, tmaxs;
        for(int i = 0; i < 3; i++) {
            float d = 1 / ray.heading()[i];
            float loc = ray.location()[i];
            
            float t0 = (minBox[i] - loc) * d;
            float t1 = (maxBox[i] - loc) * d;

            if(d < 0) {
                tmins[i] = t1; tmaxs[i] = t0;
            } else {
                tmins[i] = t0; tmaxs[i] = t1;
            }
        }
        float tmin = tmins.max(), tmax = tmaxs.min();
        if(tmin <= tmax && tmax > 0) {
            return tmin;
        } else {
            return maxDist;
        }
    }
};

class BVHTree{
public:

    uint nodeCount, targetCount;
    Node* nodes;
    Target** targets;
    uint* tIdx;

    __device__ BVHTree(targetList** listptr) : nodeCount(0), targetCount(0) {
        targetCount = (*listptr)->size;
        if(targetCount < 1) {
            printf("ERROR: BVH targetCount must be at least one.\n");
        }
        targets = (*listptr)->targets;
        tIdx = new uint[targetCount];
        for(int i = 0; i < targetCount; i++) {
            tIdx[i] = i;
        }
        nodes = new Node[2*targetCount - 1];
        Node* root = nodes;
        root->targetCount = targetCount;
        root->first = 0;
        setNodeBounds(0);
        nodeCount++;
        if(targetCount == 1) {
            return;
        }
        subDivision();
        printf("BVH tree built, including %d nodes for %d targets\n", nodeCount, targetCount);
    }

    __device__ void findCollision(const Ray& ray, HitInfo& hit) const {
        Node* stack[100];
        uint stackPtr = 0;
        Node* node = nodes;
        while(1) {
            if(node->targetCount > 0) {
                for(int i = 0; i < node->targetCount; i++) {
                    Target* obj = targets[tIdx[node->first + i]];
                    float t = obj->collision(ray);
                    if(t > epsilon) {
                        if(hit.t > t || hit.t < 0.0f) {
                            hit = HitInfo(ray.dir, ray.at(t), obj->normal(ray.at(t)), t, obj);
                        }
                    }
                }
                if(stackPtr == 0) break;
                node = stack[--stackPtr];
                continue;
            }
            Node* child1 = nodes + node->first;
            Node* child2 = nodes + node->first + 1;
            float d1 = child1->getsHit(ray), d2 = child2->getsHit(ray);
            if(d1 > d2) {
                float d = d1; d1 = d2; d2 = d;
                Node* n = child1; child1 = child2; child2 = n;
            }
            if(d1 >= maxDist) {
                if(stackPtr == 0) break;
                node = stack[--stackPtr];
            } else {
                node = child1;
                if(d2 < maxDist) stack[stackPtr++] = child2;
            }
        }
        
    }

    __device__ void clear() {
        delete[] nodes;
        delete[] tIdx;
    }

private:

    __device__ void setNodeBounds(uint nodeIdx) {
        Node* node = nodes + nodeIdx;
        Target* p = targets[tIdx[node->first]];
        node->minBox = p->minBox(); node->maxBox = p->maxBox();
        for(uint i = 1; i < node->targetCount; i++) {
            p = targets[tIdx[node->first + i]];
            node->minBox = minVector(node->minBox, p->minBox());
            node->maxBox = maxVector(node->maxBox, p->maxBox());
        }
    }

    __device__ void subDivision() {
        for(int n = 0; n < nodeCount; n++) {
            Node* node = nodes + n;
            int ax;
            float split;
            simpleSplit(node, ax, split);

            uint i = node->first, j = i + node->targetCount - 1;
            while(i <= j && j != 0) {
                if(targets[tIdx[i]]->centroid()[ax] < split) {
                    i++;
                } else {
                    uint s = tIdx[i];
                    tIdx[i] = tIdx[j];
                    tIdx[j] = s;
                    j--;
                }
            }
            uint lTargets = i - node->first;
            if(lTargets < 1 || lTargets == node->targetCount) {
                continue;
            }
            uint lNodeIdx = nodeCount++, rNodeIdx = nodeCount++;
            nodes[lNodeIdx].first = node->first;
            nodes[lNodeIdx].targetCount = lTargets;
            nodes[rNodeIdx].first = i;
            nodes[rNodeIdx].targetCount = node->targetCount - lTargets;
            node->first = lNodeIdx;
            node->targetCount = 0;

            setNodeBounds(lNodeIdx);
            setNodeBounds(rNodeIdx);
        }
    }

    __device__ void simpleSplit(Node* node, int& axis, float& splitPosition) {
        Vector3D diag = node->maxBox - node->minBox;
        axis = (diag.x > diag.y) ? (diag.x > diag.z ? 0 : 2) : (diag.y > diag.z ? 1 : 2);
        splitPosition = node->minBox[axis] + diag[axis] * 0.5f;
    }
};

#endif