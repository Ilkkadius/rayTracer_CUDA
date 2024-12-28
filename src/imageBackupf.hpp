#ifndef IMAGE_BACKUP_CUDA_HPP
#define IMAGE_BACKUP_CUDA_HPP

#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>

#include "logMethods.hpp"
#include "auxiliaryf.hpp"

namespace Backup{

    static std::mutex fileMutex;
    /**
     * @brief Writes a single column of pixels to a .txt file given by "path"
     * 
     * @param path 
     * @param pixels 
     * @param column 
     * @param width 
     * @param height 
     */
    void imageToText(const std::string& path, sf::Uint8* pixels, int column, int width, int height);

    __host__ void fullImageToText(const std::string& path, sf::Uint8* pixels, int width, int height, int samples = 0);

    /**
     * @brief Store half of a picture at a time, dictated by boolean left
     * 
     * @param path 
     * @param pixels 
     * @param width 
     * @param height 
     * @param left , true equals to a range of [0, width/2-1], false equals to [width/2, width]
     */
    __host__ void halfImageToText(const std::string& path, sf::Uint8* pixels, int width, int height, bool left);


    __host__ bool textToImage(const std::string& path, int width, int height);

    static std::mutex binaryMutex;
    /**
     * @brief Write a complete column to a binary file in "path". Generates additional .txt-file for saving the column order.
     * 
     * @param path      path to the binary file (.bin), title must contain the image dimensions separated by 'x', e.g. 1920x1080
     * @param pixels    Array containing the pixel RGB and alpha -values
     * @param x         index of column
     * @param width     image width
     * @param height    image height
     */
    void imageToBinary(const std::string& path, sf::Uint8* pixels, int x, int width, int height);

    bool binaryToImage(const std::string& binary, const std::string& order);

    void fullImageToBinary(sf::Uint8* pixels, int width, int height);

    bool fullBinaryToImage(const std::string& binary);

    void halfImageToBinary(const std::string& path, sf::Uint8* pixels, int width, int height, bool left);

    void quarterImageToBinary(const std::string& path, sf::Uint8* pixels, int width, int height, int quarter);

    void appendOrderedBinaries(const std::string& first, const std::string& second);

}

#endif