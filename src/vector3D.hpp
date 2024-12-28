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

    __host__ __device__ Vector3D& operator=(const Vector3D& vec);

    __host__ __device__ Vector3D& operator+=(const Vector3D& vec);

    __host__ __device__ Vector3D& operator-=(const Vector3D& vec);

    __host__ __device__ Vector3D operator-() const;

    __host__ __device__ float lengthSquared() const;

    __host__ __device__ float length() const;

    __host__ __device__ float norm() const;

    /**
     * @brief Give the value of the maximum component of the Vector3D
     * 
     * @return float
     */
    __host__ __device__ float max() const;

    /**
     * @brief Give the value of the minimum component of the Vector3D
     * 
     * @return float
     */
    __host__ __device__ float min() const;

    __host__ __device__ float operator[](int i) const;

    __host__ __device__ float& operator[](int i);


}; //##############################################

__host__ std::ostream& operator<<(std::ostream& os, const Vector3D& vec);

/**
 * @brief Calculates the dot product between two given vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return float, the result of the dot product
 */
__host__ __device__ float Dot(const Vector3D& vec1, const Vector3D& vec2);

/**
 * @brief Calculates the cross product between two given vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D, the result of the cross product
 */
__host__ __device__ Vector3D Cross(const Vector3D& vec1, const Vector3D& vec2);

/**
 * @brief Creates a vector from the component-wise product of two vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(const Vector3D& vec1, const Vector3D& vec2);

/**
 * @brief Calculate the sum of two vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator+(const Vector3D& vec1, const Vector3D& vec2);

/**
 * @brief Calculate the difference between two vectors
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator-(const Vector3D& vec1, const Vector3D& vec2);

/**
 * @brief Calculate a vector-scalar product
 * 
 * @param t 
 * @param vec 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(float t, const Vector3D& vec);

/**
 * @brief Calculate a vector-scalar product
 * 
 * @param vec 
 * @param t 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator*(const Vector3D& vec, float t);

/**
 * @brief Divide a vector by a scalar
 * 
 * @param vec 
 * @param t 
 * @return Vector3D 
 */
__host__ __device__ Vector3D operator/(const Vector3D& vec, float t);

/**
 * @brief Create a unit vector from a given vector
 * 
 * @param vec 
 * @return Vector3D, the unit vector
 */
__host__ __device__ Vector3D unitVec(const Vector3D& vec);

/**
 * @brief Construct a new vector from the minimum components of each given vector
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D minVector(const Vector3D& vec1, const Vector3D& vec2);

/**
 * @brief Construct a new vector from the maximum components of each given vector
 * 
 * @param vec1 
 * @param vec2 
 * @return Vector3D 
 */
__host__ __device__ Vector3D maxVector(const Vector3D& vec1, const Vector3D& vec2);



#endif