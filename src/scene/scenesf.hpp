#ifndef SCENES_CUDA_HPP
#define SCENES_CUDA_HPP

#include "cuda_runtime.h"

#include "compoundf.hpp"

enum class sceneType {
    EMPTY,
    PLATON,
    CSG,
    LIGHT,
    TEST,
};

__host__ inline std::ostream& operator<<(std::ostream& os, sceneType type) {
    switch(type) {
        case sceneType::EMPTY:
            os << "EMPTY";
            break;
        case sceneType::PLATON:
            os << "PLATON";
            break;
        case sceneType::CSG:
            os << "CSG";
            break;
        case sceneType::LIGHT:
            os << "LIGHT";
            break;
        case sceneType::TEST:
            os << "TEST";
            break;
    }
    return os;
}

namespace Scene{

    __host__ bool parseScene(std::string line, sceneType& type);

    __device__ void testScene(TargetList* list);

    __device__ void empty(TargetList* list);

    __device__ void Platon(TargetList* list);

    __device__ void CSG(TargetList* list);

    __device__ void light(TargetList* list);
};

#endif