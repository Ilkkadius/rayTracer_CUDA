#ifndef AUXILIARY_CUDA_HPP
#define AUXILIARY_CUDA_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "vector3D.hpp"

#include "macros.hpp"


static constexpr float epsilon = 0.0001f; // Do not decrease, shadow acne will occur
static constexpr float phi = 1.61803f;
typedef unsigned int uint;
typedef std::chrono::system_clock::time_point timepoint;

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

    __host__ void uppercase(std::string& s);
    __host__ bool stringToInt(const std::string& str, int& num);

    std::string zero2front(int a);

    __host__ inline std::string parentDirectory(const char* path) {
		std::string maincpp = std::string(path); int spot = 0;
		for(int i = 0; i < maincpp.size(); i++)
			if(maincpp[i] == '/' || maincpp[i] == '\\')
				spot = i;
        return maincpp.substr(0, spot) + "/";
    }

    /**
     * @brief e.g. 1. Jan. 2023: 20:30:15
     */
    std::string getDate();

    int getMonthNumber(const std::string& month);

    /**
     * @brief e.g. 231231_235932
     */
    std::string getRawDate();

    /**
     * @brief e.g. 7min34s
     */
    std::string getRawDuration(double duration, int precision = 2);

    std::string getDuration(double duration);

    inline std::string line(int width) {
        std::stringstream ss;
        for(int i = 0; i < width; i++)
            ss << "-";
        return ss.str();
    }

    inline void error(const std::string& message) {
        int l = std::min(50, (int)message.size());
        std::cout << "\n" << aux::line(l) << std::endl;
        std::cout << message << std::endl;
        std::cout << aux::line(l) << std::endl;
        exit(1);
    }
    inline void error(const char* filepath, uint linenum, const std::string& message) {
        std::cout << "\n" << aux::line(50) << std::endl;
        std::cout << "ERROR in " << filepath << std::endl;
        std::cout << "Line " << linenum << ": " << message << std::endl;
        std::cout << aux::line(50) << std::endl;
        exit(1);
    }

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