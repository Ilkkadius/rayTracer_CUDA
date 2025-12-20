#ifndef IMAGE_BACKUP_CUDA_HPP
#define IMAGE_BACKUP_CUDA_HPP

#include <SFML/Graphics.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>

#include "logMethods.hpp"

namespace Backup{

    static std::mutex binaryMutex;

    void imageToBinary(sf::Uint8* pixels, int width, int height);

    bool binaryToImage(const std::string& binaryFile);

}

#endif