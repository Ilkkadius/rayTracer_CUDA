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

std::string zero2front(int a)
{
    return a >= 10 ? "" : "0";
}

std::string aux::getDate() {
    auto t = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(t);
    tm* timeInfo = localtime(&tt);
    char buffer[24];
    const std::string format = "%e. %h. %Y: %T";
    strftime(buffer, sizeof(buffer), format.c_str(), timeInfo);
    return std::string(buffer);
}

int aux::getMonthNumber(const std::string& month) {
    if(month[2] == 'n') {
        return (month[1] == 'a' ? 1 : 6);
    } else if(month[2] == 'b') {
        return 2;
    } else if(month[2] == 'r') {
        return (month[0] == 'M' ? 3 : 4);
    } else if(month[2] == 'y') {
        return 5;
    } else if(month[2] == 'l') {
        return 7;
    } else if(month[2] == 'g') {
        return 8;
    } else if(month[2] == 'p') {
        return 9;
    } else if(month[2] == 't') {
        return 10;
    } else if(month[2] == 'v') {
        return 11;
    } else {
        return 12;
    }
}

std::string aux::getRawDate() {
    auto t = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(t);
    tm* timeInfo = localtime(&tt);
    char buffer[13];
    const std::string format = "%y%m%d_%H%M";
    strftime(buffer, sizeof(buffer), format.c_str(), timeInfo);
    return std::string(buffer);
}

std::string aux::getRawDuration(double duration, int precision) {
    std::stringstream ss;
    int minutes = 0;
    if(duration > 60) {
        minutes = duration / 60;
        ss << minutes << "min";
    }
    ss << std::fixed << std::setprecision(precision) << duration - 60.0 * minutes << "s";
    return ss.str();
}

std::string aux::getDuration(double duration) {
    int minutes = 0;
    std::stringstream runtime;
    if (duration > 60.0)
    {
        minutes = duration / 60;
        runtime << minutes << " minutes ";
    }
    runtime << duration - 60.0 * minutes << " seconds.";
    return runtime.str();
}

