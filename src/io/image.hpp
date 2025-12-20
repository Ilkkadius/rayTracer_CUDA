#ifndef IMAGEPROCESSES_CUDA_HPP
#define IMAGEPROCESSES_CUDA_HPP

#include <SFML/Graphics.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>

#include "logMethods.hpp"
#include "termcolor.hpp"

namespace Image {

    void writePNG(sf::Uint8* pixels, int width, int height, int samples, double duration);

    static std::mutex binaryMutex;

    void ToBinary(sf::Uint8* pixels, int width, int height);

    bool fromBinary(const std::string& binaryFile);

};

#endif