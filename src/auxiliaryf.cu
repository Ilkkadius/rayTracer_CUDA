#include "auxiliaryf.hpp"


__device__ float aux::randUnitFloat(curandState *state) {
    return 2.0f*curand_uniform(state) - 1.0f;
}

__device__ float aux::randFloat(curandState *state, float lower, float upper) {
    return (upper - lower)*curand_uniform(state) - lower;
}

__device__ Vector3D aux::randUnitVec(curandState *state) {
    Vector3D vec(0.0f, 0.0f, 0.0f);
    float squareLength = 2.0f;
    while(squareLength > 1.0f) {
        vec = Vector3D(randUnitFloat(state), randUnitFloat(state), randUnitFloat(state));
        squareLength = vec.lengthSquared();
    }
    return vec / sqrtf(squareLength);
}

__device__ Vector3D aux::randHemisphereVec(curandState *state, const Vector3D& normal) {
    Vector3D vec = randUnitVec(state);
    if(Dot(normal, vec) < epsilon) {
        vec = -vec;
    }
    return vec;
}

__host__ void aux::uppercase(std::string& s) {
    auto upper = [](char c) {return std::toupper(c);};
    std::transform(s.begin(), s.end(), s.begin(), upper);
}

__host__ bool aux::stringToInt(const std::string& str, int& num) {
    char* p;
    float t = std::strtod(str.c_str(), &p);
    if(*p) {
        return false;
    }
    num = t;
    return true;
}
