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

namespace Backup{

    std::mutex fileMutex;
    __host__ void imageToFile(const char* path, sf::Uint8* pixels, int column, int width, int height) {
        std::lock_guard<std::mutex> lock(fileMutex);
        std::ofstream file(path, std::ofstream::app);
        if(file.is_open()) {
            int r = 0, idx = 4*column;
            file << "#" << column << std::endl;
            while(r < height) {
                file << 
                    static_cast<int>(pixels[idx]) << " " << 
                    static_cast<int>(pixels[idx + 1]) << " " << 
                    static_cast<int>(pixels[idx +2]) << std::endl;
                r++; idx += 4*width;
            }
            file.close();
        } else {
            std::cout << "\033[31mColumn " << column << "was not saved.\033[0m" << std::endl;
        }
    }

    __host__ void imageToFile(const char* path, sf::Uint8* pixels, int width, int height) {
        for(int i = 0; i < width; i++) {
            imageToFile(path, pixels, i, width, height);
        }
    }

    __host__ void fileToImage(const char* path, int width, int height) {
        std::ifstream is(path);
        if(!is.is_open()) {
            std::cout << "\033[31mError in opening the file\033[0m" << std::endl;
        }

        sf::Uint8 pixels[width*height*4];
        std::string line;
        int row = 0, column = -1;
        while(std::getline(is, line)) {
            if(line[0] == '#') {
                std::stringstream ss(line.substr(1));
                ss >> column;
                row = 0;
            } else if(column >= 0 && row < height) {
                uint r = 0, g = 0, b = 0;
                std::stringstream ss(line);
                ss >> r >> g >> b;
                r = r < 256 ? r : 255;
                g = g < 256 ? g : 255;
                b = b < 256 ? b : 255;
                uint idx = 4*(row*width + column);
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
                row++;
            }
        }
        is.close();


        sf::Texture texture;
        texture.create(width, height);
        sf::Sprite sprite(texture);
        texture.update(pixels);
        sf::Image image = texture.copyToImage();
        image.saveToFile("" + getRawDate() + "_backup.png");
    }

}

#endif