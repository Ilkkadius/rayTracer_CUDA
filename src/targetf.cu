#include "targetf.hpp"

__device__ Target::Target(Shape* shape_, const Vector3D& color_, float emissivity_) : shape(shape_), color(color_), emissivity(emissivity_) {}

__device__ float Target::collision(const Ray& ray) const {
    return shape->rayCollision(ray);
}

__device__ Vector3D Target::normal(const Vector3D& point) const {
    return shape->normal(point);
}

__device__ Vector3D Target::centroid() const {
    return shape->centroid();
}

__device__ void Target::translate(const Vector3D& vec) {shape->translate(vec);}
__device__ void Target::translate(float x, float y, float z) {shape->translate(x,y,z);}

__device__ void Target::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
    shape->rotate(angle, axis, axisPos);
}

__device__ bool Target::isRadiant() const {
    return emissivity > 0.0f;
}

__device__ Vector3D Target::emission() const {
    return emissivity*color;
}

__device__ Vector3D Target::minBox() const {
    return shape->minBox;
}

__device__ Vector3D Target::maxBox() const {
    return shape->maxBox;
}


