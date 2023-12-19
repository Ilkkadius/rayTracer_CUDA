#ifndef LOGMETHODS_CUDA_HPP
#define LOGMETHODS_CUDA_HPP

#include "cuda_runtime.h"

#include <iostream>
#include <sstream>

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
    date << dd << ". " << mm << ". " << yy << ": " << zero2front(h) << h << ":" << zero2front(m) << m << ":" << zero2front(s) << s << std::endl;
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

    

#endif