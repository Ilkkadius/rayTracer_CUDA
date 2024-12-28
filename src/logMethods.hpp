#ifndef LOGMETHODS_CUDA_HPP
#define LOGMETHODS_CUDA_HPP

#include "cuda_runtime.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

__host__ std::string zero2front(int a);

/**
 * @brief e.g. 1. Jan. 2023: 20:30:15
 */
__host__ std::string getDate();

__host__ int getMonthNumber(const std::string& month);

/**
 * @brief e.g. 231231_2359
 */
__host__ std::string getRawDate();

/**
 * @brief e.g. 7min34s
 */
std::string getRawDuration(double duration, int precision = 2);

std::string getDuration(double duration);

/**
 * @brief e.g. 1920x1080
 * 
 */
std::string getImageDimensions(int width, int height);

__host__ std::string getImageFilename(int width, int height, int samples, double duration);

    

#endif