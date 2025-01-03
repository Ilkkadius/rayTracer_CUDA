#ifndef AUXILIARY_CUDA_HPP
#define AUXILIARY_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vector3D.hpp"

#define M_PI 3.14159265358979323846


#define THREADS_PER_BLOCK int(512)


static constexpr float epsilon = 0.0001f; // Do not decrease, shadow acne will occur
static constexpr float phi = 1.61803f;
typedef unsigned int uint;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel;

#define CHECK_FUNC
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

#define divup_FUNC

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

namespace aux{

    /**
     * @brief Random float between [-1.0f, 1.0f]
     * 
     * @param state 
     * @return __device__ 
     */
    __device__ float randUnitFloat(curandState *state);

    __device__ float randFloat(curandState *state, float lower, float upper);

    __device__ Vector3D randUnitVec(curandState *state);

    __device__ Vector3D randHemisphereVec(curandState *state, const Vector3D& normal);

}

    template <typename T>
    class dynVec{
    public:
        
        __host__ __device__ dynVec(int reserve) : size_(0), capacity_(reserve), arr(new T[reserve*sizeof(T)]) {}

        __host__ __device__ void push_back(T elem) {
            if(size_ >= capacity_) {
                grow();
            }
            arr[size_++] = elem;
        }

        __host__ __device__ ~dynVec() {
            delete[] arr;
        }

        __host__ __device__ T operator[](int i) const {return arr[i];}

        __host__ __device__ size_t size() const {return size_;}

        __host__ __device__ size_t capacity() const {return capacity_;}

        __host__ __device__ const T* getArray() {
            return arr;
        }

    private:
        size_t size_, capacity_;
        T* arr;

        __host__ __device__ void grow() {
            T* big = new T[capacity_*sizeof(T) << 1];
            capacity_ = capacity_ << 1;

            for(int i = 0; i < size_; i++) {
                big[i] = arr[i];
            }

            delete[] arr;
            arr = big;
        }
    };


#endif