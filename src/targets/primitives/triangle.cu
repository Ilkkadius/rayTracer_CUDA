#include "triangle.hpp"

__device__ Triangle::Triangle(const Vector3D& vertex0, const Vector3D& vertex1, const Vector3D& vertex2, const Vector3D& color_, float emissivity_) 
                    : v0(vertex0), v1(vertex1), v2(vertex2) {color = color_; emissivity = emissivity_;}

__device__ bool Triangle::rayCollision(const Ray& ray, HitInfo* hit) const {
    Vector3D edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = Cross(ray.heading(), edge2);
    a = Dot(edge1, h);

    if (a > -epsilon && a < epsilon)
        return false;

    f = 1.0f / a;
    s = ray.location() - v0;
    u = f * Dot(s,h);

    if (u < 0.0f || u > 1.0f)
        return false;

    q = Cross(s, edge1);
    v = f * Dot(ray.heading(), q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = f * Dot(edge2, q);

    if (t > epsilon) {
        h = unitVec(Cross(edge1,edge2));
        hit->t = t;
        hit->color = color;
        hit->normal = Dot(h,ray.dir) > 0.0f ? -h : h;
        hit->emission = emissivity;
        return true;
    }
    return false;
}

__device__ Vector3D Triangle::centroid() const {
    return (v0 + v1 + v2)/3.0f;
}

__device__ void Triangle::translate(const Vector3D& vec) {
    v0 += vec; v1 += vec; v2 += vec;
}

__device__ void Triangle::rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
    v0 = rotateVec(v0, angle, axis, axisPos);
    v1 = rotateVec(v1, angle, axis, axisPos);
    v2 = rotateVec(v2, angle, axis, axisPos);
}

__device__ Vector3D Triangle::minBox() const {return minVector(v2, minVector(v0, v1));}
__device__ Vector3D Triangle::maxBox() const {return maxVector(v2, maxVector(v0, v1));}
