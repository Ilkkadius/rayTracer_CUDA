#ifndef CONSTRUCTIVESOLIDGEOMETRY_CUDA_HPP
#define CONSTRUCTIVESOLIDGEOMETRY_CUDA_HPP

#include "target.hpp"

enum class CSG {
    UNION,
    DIFFERENCE,
    INTERSECTION,
    NONE
};

struct CSGNode {
    uint first, maxCollisionCount;
    CSG oper;
};

class ConstructiveShape : public Target {
public:
    Target** targets;
    CSGNode* nodes;
    uint* nodePostorder;

    uint targetCount;
    uint nodeCount;

    __device__ ConstructiveShape(Target** targetList, uint numTargets, CSGNode* nodeList, uint numNodes);

    __device__ bool rayCollision(const Ray& ray, HitInfo* hit) const;
    __device__ int allCollisions(const Ray& ray, HitInfo* hitlist) const;
    __device__ int maxCollisions() const {return nodes[0].maxCollisionCount;}

    __device__ Vector3D centroid() const;
    
    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

    __device__ Vector3D emission() const;

    __device__ Vector3D minBox() const;
    __device__ Vector3D maxBox() const;

    __device__ virtual void release() {
        if(targets) {
            for(int i = 0; i < targetCount; i++) {
                delete targets[i];
            }
            delete[] targets;
        }
        if(nodes) {
            delete[] nodes; delete[] nodePostorder;
        }
    };

private:

    __device__ void handleHits(const CSGNode& node, int lastHitIdx, HitInfo* hitlist, HitInfo* aux)const;

    __device__ void buildPostorder() {
        nodePostorder = new uint[nodeCount];
        int stack[100], current = 0;
        int lastVisit = -1, stackPtr = 0, orderPtr = 0;

        int iter = 0;

        while((stackPtr > 0) || (nodes[current].oper != CSG::NONE)) {
            iter++;
            if(nodes[current].oper != CSG::NONE) {
                stack[stackPtr++] = current;
                current = nodes[current].first;
            } else {
                CSGNode peek = nodes[stack[stackPtr-1]];

                if(lastVisit != peek.first + 1) {
                    if(current == peek.first && nodes[current].oper == CSG::NONE) {
                        nodePostorder[orderPtr++] = current;
                        nodes[current].maxCollisionCount = targets[nodes[current].first]->maxCollisions();
                    }
                    current = peek.first + 1;
                    lastVisit = current;
                    
                } else {
                    
                    if(nodes[lastVisit].oper == CSG::NONE) {
                        nodePostorder[orderPtr++] = lastVisit;
                        nodes[lastVisit].maxCollisionCount = targets[nodes[lastVisit].first]->maxCollisions();
                    }
                    nodePostorder[orderPtr++] = stack[--stackPtr];
                    lastVisit = stack[stackPtr];
                    nodes[lastVisit].maxCollisionCount = nodes[nodes[lastVisit].first].maxCollisionCount + nodes[nodes[lastVisit].first + 1].maxCollisionCount;
                }
            }
            if(iter > 2*nodeCount) {
                printf("ERROR: CSG POSTORDERING FAILED\n"); return;
            }
        }

    };

};

#endif