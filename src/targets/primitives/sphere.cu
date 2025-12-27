#include "sphere.hpp"

__device__ Sphere::Sphere(const Vector3D& center_, float radius_, const Vector3D& color_, float emissivity_) : center(center_), radius(radius_) {
    color = color_; emissivity = emissivity_;
}

__device__ bool Sphere::rayCollision(const Ray& ray, HitInfo* hit) const {
    Vector3D v = ray.pos - center;
    float vd = Dot(v, ray.dir);
    float disc = vd * vd - (v.lengthSquared() - radius * radius);
    if(disc < 0.0f) {
        return false;
    }
    float root = sqrtf(disc);
    float res = -vd - root;
    if(res < 0.0f) {
        res = -vd + root;
        if(res < 0.0f) {
            return false;
        }
    }
    hit->t = res;
    hit->normal = unitVec(ray.at(res) - center);
    hit->color = color;
    hit->emission = emissivity;
    return true;        
}
__device__ int Sphere::allCollisions(const Ray& ray, HitInfo* hitlist) const {
    Vector3D v = ray.pos - center;
    float vd = Dot(v, ray.dir);
    float disc = vd * vd - (v.lengthSquared() - radius * radius);
    if(disc < 0.0f) {
        hitlist[0].emission = -1.0f;
        hitlist[1].emission = -1.0f;
        return 0;
    }
    float root = sqrtf(disc);
    float res = -vd - root;
    if(res < 0.0f) {
        res = -vd + root;
        if(res < 0.0f) {
            hitlist[0].emission = -1.0f;
            hitlist[1].emission = -1.0f;
            return 0;
        }
    }

    hitlist[0].t = -vd - root;
    hitlist[0].color = color;
    hitlist[0].emission = emissivity;
    hitlist[0].normal = unitVec(ray.at(-vd - root) - center);

    hitlist[1].t = -vd + root;
    hitlist[1].color = color;
    hitlist[1].emission = emissivity;
    hitlist[1].normal = unitVec(ray.at(-vd + root) - center);
    return 2;     
}

__device__ Vector3D Sphere::centroid() const {return center;}

__device__ void Sphere::translate(const Vector3D& vec) {
    center += vec;
}

__device__ void Sphere::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
    center = rotateVec(center, angle, axis, axisPos);
}

__device__ void Sphere::affine(const Matrix& A, const Vector3D& b) {
    center = A * center + b;
}

__device__ Vector3D Sphere::minBox() const {return center - Vector3D(radius, radius, radius);}
__device__ Vector3D Sphere::maxBox() const {return center + Vector3D(radius, radius, radius);}