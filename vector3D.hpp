#ifndef LINEAR_ALGEBRA_CUDA_HPP
#define LINEAR_ALGEBRA_CUDA_HPP

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

class Vector3D {
public:
    float x, y, z;

    // Default constructor
    __host__ __device__ Vector3D() : x(0.0f), y(0.0f), z(0.0f) {}

    // Constructor
    __host__ __device__ Vector3D(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    // Copy constructor
    __host__ __device__ Vector3D(const Vector3D& vec) : x(vec.x), y(vec.y), z(vec.z) {}

    __host__ __device__ Vector3D& operator=(const Vector3D& vec) {
        x = vec.x; y = vec.y; z = vec.z;
        return *this;
    }

    __host__ __device__ Vector3D& operator+=(const Vector3D& vec) {
        x += vec.x; y += vec.y; z += vec.z;
        return *this;
    }

    __host__ __device__ Vector3D& operator-=(const Vector3D& vec) {
        x -= vec.x; y -= vec.y; z -= vec.z;
        return *this;
    }

    __host__ __device__ Vector3D operator-() const {
        return Vector3D(-x, -y, -z);
    }

    __host__ __device__ float lengthSquared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ float length() const {
        return sqrtf(lengthSquared());
    }

    __host__ __device__ float norm() const {
        return length();
    }

    /**
     * @brief Give the value of the maximum component of the Vector3D
     * 
     * @return float
     */
    __host__ __device__ float max() const {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }

    /**
     * @brief Give the value of the minimum component of the Vector3D
     * 
     * @return float
     */
    __host__ __device__ float min() const {
        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }


}; //##############################################

/**
 * @brief Calculates the dot product between two given vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return float, the result of the dot product
 */
__host__ __device__ float Dot(const Vector3D& vec1, const Vector3D& vec2) {
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

/**
 * @brief Calculates the cross product between two given vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D, the result of the cross product
 */
__host__ __device__ Vector3D Cross(const Vector3D& vec1, const Vector3D& vec2) {
    float x = vec1.y * vec2.z - vec1.z * vec2.y;
    float y = -(vec1.x * vec2.z - vec1.z * vec2.x);
    float z = vec1.x * vec2.y - vec1.y * vec2.x;
    return Vector3D(x, y, z);
}

/**
 * @brief Creates a vector from the component-wise product of two vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(const Vector3D& vec1, const Vector3D& vec2) {
    return Vector3D(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z);
}

/**
 * @brief Calculate the sum of two vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator+(const Vector3D& vec1, const Vector3D& vec2) {
    return Vector3D(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z);
}

/**
 * @brief Calculate the difference between two vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator-(const Vector3D& vec1, const Vector3D& vec2) {
    return Vector3D(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z);
}

/**
 * @brief Calculate a vector-scalar product
 * 
 * @param t 
 * @param vec 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(float t, const Vector3D& vec) {
    return Vector3D(t*vec.x, t*vec.y, t*vec.z);
}

/**
 * @brief Calculate a vector-scalar product
 * 
 * @param vec 
 * @param t 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(const Vector3D& vec, float t) {
    return t*vec;
}

/**
 * @brief Divide a vector by a scalar
 * 
 * @param vec 
 * @param t 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator/(const Vector3D& vec, float t) {
    return (1.0f/t)*vec;
}

/**
 * @brief Create a unit vector from a given vector
 * 
 * @param vec 
 * @return Vector3D, the unit vector
 */
__host__ __device__ Vector3D unitVec(const Vector3D& vec) {
    return vec/vec.length();
}


#endif