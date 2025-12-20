#include "vector3D.hpp"


__host__ __device__ Vector3D& Vector3D::operator=(const Vector3D& vec) {
    x = vec.x; y = vec.y; z = vec.z;
    return *this;
}

__host__ __device__ Vector3D& Vector3D::operator+=(const Vector3D& vec) {
    x += vec.x; y += vec.y; z += vec.z;
    return *this;
}

__host__ __device__ Vector3D& Vector3D::operator-=(const Vector3D& vec) {
    x -= vec.x; y -= vec.y; z -= vec.z;
    return *this;
}

__host__ __device__ Vector3D Vector3D::operator-() const {
    return Vector3D(-x, -y, -z);
}

__host__ __device__ float Vector3D::lengthSquared() const {
    return x * x + y * y + z * z;
}

__host__ __device__ float Vector3D::length() const {
    return sqrtf(lengthSquared());
}

__host__ __device__ float Vector3D::norm() const {
    return length();
}

/**
 * @brief Give the value of the maximum component of the Vector3D
 * 
 * @return float
 */
__host__ __device__ float Vector3D::max() const {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

/**
 * @brief Give the value of the minimum component of the Vector3D
 * 
 * @return float
 */
__host__ __device__ float Vector3D::min() const {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

__host__ __device__ float Vector3D::operator[](int i) const {
    switch(i) {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        default:
            return z;
            break;
    }
}

__host__ __device__ float& Vector3D::operator[](int i) {
    switch(i) {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        default:
            return z;
            break;
    }
}


__host__ std::ostream& operator<<(std::ostream& os, const Vector3D& vec) {
    os << "(" << vec.x << "; " << vec.y << "; " << vec.z << ")";
    return os;
}

__host__ __device__ float Dot(const Vector3D& vec1, const Vector3D& vec2) {
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__host__ __device__ Vector3D Cross(const Vector3D& vec1, const Vector3D& vec2) {
    float x = vec1.y * vec2.z - vec1.z * vec2.y;
    float y = -(vec1.x * vec2.z - vec1.z * vec2.x);
    float z = vec1.x * vec2.y - vec1.y * vec2.x;
    return Vector3D(x, y, z);
}

__host__ __device__ Vector3D operator*(const Vector3D& vec1, const Vector3D& vec2) {
    return Vector3D(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z);
}

__host__ __device__ Vector3D operator+(const Vector3D& vec1, const Vector3D& vec2) {
    return Vector3D(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z);
}

__host__ __device__ Vector3D operator-(const Vector3D& vec1, const Vector3D& vec2) {
    return Vector3D(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z);
}

__host__ __device__ Vector3D operator*(float t, const Vector3D& vec) {
    return Vector3D(t*vec.x, t*vec.y, t*vec.z);
}

__host__ __device__ Vector3D operator*(const Vector3D& vec, float t) {
    return t*vec;
}

__host__ __device__ Vector3D operator/(const Vector3D& vec, float t) {
    return (1.0f/t)*vec;
}

__host__ __device__ Vector3D unitVec(const Vector3D& vec) {
    return vec/vec.length();
}

__host__ __device__ Vector3D minVector(const Vector3D& vec1, const Vector3D& vec2) {
    float x = vec1.x < vec2.x ? vec1.x : vec2.x;
    float y = vec1.y < vec2.y ? vec1.y : vec2.y;
    float z = vec1.z < vec2.z ? vec1.z : vec2.z;
    return Vector3D(x, y, z);
}

__host__ __device__ Vector3D maxVector(const Vector3D& vec1, const Vector3D& vec2) {
    float x = vec1.x > vec2.x ? vec1.x : vec2.x;
    float y = vec1.y > vec2.y ? vec1.y : vec2.y;
    float z = vec1.z > vec2.z ? vec1.z : vec2.z;
    return Vector3D(x, y, z);
}
