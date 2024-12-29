#ifndef LOGMETHODS_CUDA_HPP
#define LOGMETHODS_CUDA_HPP

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

std::string zero2front(int a);

/**
 * @brief e.g. 1. Jan. 2023: 20:30:15
 */
std::string getDate();

int getMonthNumber(const std::string& month);

/**
 * @brief e.g. 231231_2359
 */
std::string getRawDate();

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

std::string getImageFilename(int width, int height, int samples, double duration);

    

#endif