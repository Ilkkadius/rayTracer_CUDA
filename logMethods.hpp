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

__host__ std::string getDate() {
    int h, m, s, dd, yy;
    char d;
    std::string mm;
    std::stringstream time(__TIME__);
    std::stringstream date(__DATE__);

    time >> h >> d >> m >> d >> s;
    date >> mm >> dd >> yy;

    std::stringstream().swap(date);
    date << dd << ". " << mm << ". " << yy << ": " << zero2front(h) << h << ":" << zero2front(m) << m << ":" << zero2front(s) << s;
    //dd. <Month>. yyyy: hh:mm:ss
    return date.str();
}

__host__ std::string getRawDate() {
    int h, m, s, dd, yy;
    char d;
    std::string mm;
    std::stringstream time(__TIME__);
    std::stringstream date(__DATE__);

    time >> h >> d >> m >> d >> s;
    date >> mm >> dd >> yy;

    std::stringstream rawDate;
    rawDate << zero2front(dd) << dd << mm << yy - 2000 << "_" << zero2front(h) << h << zero2front(m) << m;
    return rawDate.str();
}

__host__ std::string getDuration(double duration, int precision = 2) {
    std::stringstream ss;
    int minutes = 0;
    if(duration > 60) {
        minutes = duration / 60;
        ss << minutes << "min";
    }
    ss << std::fixed << std::setprecision(precision) << duration - 60.0 * minutes << "s";
    return ss.str();
}

__host__ std::string getImageFilename(int width, int height, int samples, double duration) {
    std::stringstream ss;
    ss << width << "x" << height << "_" << samples << "samples_" << getDuration(duration, 0);
    return getRawDate() + "_GPU_" + ss.str() + "_figure.png";
}

    

#endif