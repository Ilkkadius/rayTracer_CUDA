#include "CSG.hpp"

__device__ ConstructiveShape::ConstructiveShape(CSG operation, Target* l, Target* r) : oper(operation), left(l), right(r) {}

__device__ bool ConstructiveShape::rayCollision(const Ray& ray, HitInfo* hit) const {return false;}
__device__ int ConstructiveShape::allCollisions(const Ray& ray, HitInfo* hitlist) const {return 0;}

__device__ Vector3D ConstructiveShape::centroid() const {

}

__device__ void ConstructiveShape::translate(const Vector3D& vec) {

}
__device__ void ConstructiveShape::translate(float x, float y, float z) {

}

__device__ void ConstructiveShape::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {

}

__device__ Vector3D ConstructiveShape::emission() const {return left->emission();}

__device__ Vector3D ConstructiveShape::minBox() const {return minVector(left->minBox(), right->minBox());}
__device__ Vector3D ConstructiveShape::maxBox() const {return maxVector(left->maxBox(), right->maxBox());}
