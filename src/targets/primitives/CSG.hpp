#ifndef CONSTRUCTIVESOLIDGEOMETRY_CUDA_HPP
#define CONSTRUCTIVESOLIDGEOMETRY_CUDA_HPP

#include "target.hpp"

enum class CSG {
    UNION,
    DIFFERENCE,
    INTERSECTION
};

class ConstructiveShape : public Target {
public:

    __device__ ConstructiveShape(CSG operation, Target* l, Target* r);

    __device__ bool rayCollision(const Ray& ray, HitInfo* hit) const;
    __device__ int allCollisions(const Ray& ray, HitInfo* hitlist) const;

    __device__ Vector3D centroid() const;
    
    __device__ void translate(const Vector3D& vec);
    __device__ void translate(float x, float y, float z);

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos);

    __device__ virtual Vector3D emission() const;

    __device__ Vector3D minBox() const;
    __device__ Vector3D maxBox() const;

    CSG oper;
    Target* left;
    Target* right;
};

#endif