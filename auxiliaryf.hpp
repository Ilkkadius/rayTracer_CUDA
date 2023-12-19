#ifndef AUXILIARY_CUDA_HPP
#define AUXILIARY_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vector3D.hpp"

static constexpr float epsilon = 0.0001f; // Do not decrease, shadow acne will occur
static constexpr float phi = 1.61803f;

namespace aux{

    /**
     * @brief Random float between [-1.0f, 1.0f]
     * 
     * @param state 
     * @return __device__ 
     */
    __device__ float randUnitFloat(curandState *state) {
        return 2.0f*curand_uniform(state) - 1.0f;
    }

    __device__ float randFloat(curandState *state, float lower, float upper) {
        return (upper - lower)*curand_uniform(state) - lower;
    }

    __device__ Vector3D randUnitVec(curandState *state) {
        Vector3D vec(0.0f, 0.0f, 0.0f);
        float squareLength = 2.0f;
        while(squareLength > 1.0f) {
            vec = Vector3D(randUnitFloat(state), randUnitFloat(state), randUnitFloat(state));
            squareLength = vec.lengthSquared();
        }
        return vec / sqrtf(squareLength);
    }

    __device__ Vector3D randHemisphereVec(curandState *state, const Vector3D& normal) {
        Vector3D vec = randUnitVec(state);
        if(Dot(normal, vec) < epsilon) {
            vec = -vec;
        }
        return vec;
    }

}


#endif