#include "geometria.hpp"


__device__ Sphere::Sphere(const Vector3D& center_, float radius_) : center(center_), radius(radius_) {
    buildBox();
}

__device__ Vector3D Sphere::normal(const Vector3D& point) const {
    return unitVec(point - center);
}

__device__ float Sphere::rayCollision(const Ray& ray) const {
    Vector3D v = ray.pos - center;
    float vd = Dot(v, ray.dir);
    float disc = vd * vd - (v.lengthSquared() - radius * radius);
    if(disc < 0.0f) {
        return -1.0f;
    }
    float root = sqrtf(disc);
    float res = -vd - root;
    if(res < 0.0f) {
        res = -vd + root;
        if(res < 0.0f) {
            return -1.0f;
        }
    }
    return res;        
}

__device__ Vector3D Sphere::centroid() const {return center;}

__device__ void Sphere::translate(const Vector3D& vec) {
    center += vec; buildBox();
}
__device__ void Sphere::translate(float x, float y, float z) {
    translate(Vector3D(x,y,z)); buildBox();
}

__device__ void Sphere::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
    center = rotateVec(center, angle, axis, axisPos); buildBox();
}


__device__ void Sphere::buildBox() {
    Vector3D r(radius, radius, radius);
    minBox = center - r; maxBox = center + r;
}


__device__ Triangle::Triangle(const Vector3D& vertex0, const Vector3D& vertex1, const Vector3D& vertex2) : v0(vertex0), v1(vertex1), v2(vertex2) {
    buildBox();
}

__device__ Vector3D Triangle::normal(const Vector3D& point) const {
    return unitVec(Cross(v1-v0, v2-v0));
}

__device__ float Triangle::rayCollision(const Ray& ray) const {
    Vector3D edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = Cross(ray.heading(), edge2);
    a = Dot(edge1, h);

    if (a > -epsilon && a < epsilon)
        return -1.0f;

    f = 1.0f / a;
    s = ray.location() - v0;
    u = f * Dot(s,h);

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    q = Cross(s, edge1);
    v = f * Dot(ray.heading(), q);

    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;

    float t = f * Dot(edge2, q);

    if (t > epsilon) {
        return t;
    }
    return -1.0f;
}

__device__ Vector3D Triangle::centroid() const {
    return (v0 + v1 + v2)/3.0f;
}

__device__ void Triangle::translate(const Vector3D& vec) {
    v0 += vec; v1 += vec; v2 += vec; buildBox();
}
__device__ void Triangle::translate(float x, float y, float z) {
    translate(Vector3D(x,y,z)); buildBox();
}

__device__ void Triangle::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
    v0 = rotateVec(v0, angle, axis, axisPos);
    v1 = rotateVec(v1, angle, axis, axisPos);
    v2 = rotateVec(v2, angle, axis, axisPos);
    buildBox();
}


__device__ void Triangle::buildBox() {
    minBox = minVector(v2, minVector(v0, v1));
    maxBox = maxVector(v2, maxVector(v0, v1));
}
