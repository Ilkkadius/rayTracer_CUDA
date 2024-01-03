#ifndef LOGMETHODS_CUDA_HPP
#define LOGMETHODS_CUDA_HPP

#include "cuda_runtime.h"

#include <iostream>
#include <sstream>
#include <iomanip>

__host__ std::string zero2front(int a)
{
    return a >= 10 ? "" : "0";
}

/**
 * @brief e.g. 1. Jan. 2023: 20:30:15
 */
__host__ std::string getDate() {
    auto t = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(t);
    tm* timeInfo = localtime(&tt);
    char buffer[24];
    const std::string format = "%e. %h. %Y: %T";
    strftime(buffer, sizeof(buffer), format.c_str(), timeInfo);
    return std::string(buffer);
}

__host__ int getMonthNumber(const std::string& month) {
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

/**
 * @brief e.g. 231231_2359
 */
__host__ std::string getRawDate() {
    auto t = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(t);
    tm* timeInfo = localtime(&tt);
    char buffer[13];
    const std::string format = "%y%m%d_%H%M";
    strftime(buffer, sizeof(buffer), format.c_str(), timeInfo);
    return std::string(buffer);
}

/**
 * @brief e.g. 7min34s
 */
std::string getRawDuration(double duration, int precision = 2) {
    std::stringstream ss;
    int minutes = 0;
    if(duration > 60) {
        minutes = duration / 60;
        ss << minutes << "min";
    }
    ss << std::fixed << std::setprecision(precision) << duration - 60.0 * minutes << "s";
    return ss.str();
}

std::string getDuration(double duration) {
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

/**
 * @brief e.g. 1920x1080
 * 
 */
std::string getImageDimensions(int width, int height) {
    return std::to_string(width) + "x" + std::to_string(height);
}

__host__ std::string getImageFilename(int width, int height, int samples, double duration) {
    std::stringstream ss;
    ss << width << "x" << height << "_" << samples << "samples_" << getRawDuration(duration, 0);
    return getRawDate() + "_GPU_" + ss.str() + "_figure.png";
}

    

#endif