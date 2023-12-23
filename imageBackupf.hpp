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
    /**
     * @brief Writes a single column of pixels to a .txt file given by "path"
     * 
     * @param path 
     * @param pixels 
     * @param column 
     * @param width 
     * @param height 
     */
    void imageToText(const std::string& path, sf::Uint8* pixels, int column, int width, int height) {
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

    __host__ void fullImageToText(const std::string& path, sf::Uint8* pixels, int width, int height, int samples = 0) {
        for(int i = 0; i < width; i++) {
            imageToText(path, pixels, i, width, height);
        }
    }

    /**
     * @brief Store half of a picture at a time, dictated by boolean left
     * 
     * @param path 
     * @param pixels 
     * @param width 
     * @param height 
     * @param left , true equals to a range of [0, width/2-1], false equals to [width/2, width]
     */
    __host__ void halfImageToText(const std::string& path, sf::Uint8* pixels, int width, int height, bool left) {
        if(left) {
            for(int i = 0; i < width/2; i++) {
                imageToText(path, pixels, i, width, height);
            }
        } else {
            for(int i = width/2; i < width; i++) {
                imageToText(path, pixels, i, width, height);
            }
        }
    }


    __host__ bool textToImage(const std::string& path, int width, int height) {
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
        if(!image.saveToFile("" + getRawDate() + "_" + std::to_string(width) + "x" + std::to_string(height) + "_GPU_backup.png")) {
            return false;
        }
        return true;
    }



    std::mutex binaryMutex;
    /**
     * @brief Write a complete column to a binary file in "path". Generates additional .txt-file for saving the column order.
     * 
     * @param path      path to the binary file (.bin), title must contain the image dimensions separated by 'x', e.g. 1920x1080
     * @param pixels    Array containing the pixel RGB and alpha -values
     * @param x         index of column
     * @param width     image width
     * @param height    image height
     */
    void imageToBinary(const std::string& path, sf::Uint8* pixels, int x, int width, int height) {
        std::lock_guard<std::mutex> lock(fileMutex);

        std::stringstream ss;
        std::string filename(path);
        while(filename.back() != '.' && filename.length() > 1) {
            filename.pop_back();
        }
        filename.pop_back();
        ss << filename << "_columnorder.txt";

        std::ofstream order(ss.str(), std::ofstream::app);
        order << x << " ";

        std::ofstream file(path, std::ofstream::binary | std::ofstream::app);
        if(!file.is_open()) {
            std::cout << "ERROR: Binary file not open!" << std::endl;
        } else if(!order.is_open()) {
            std::cout << "ERROR: Order file not open!" << std::endl;
        } else {
            int r = 0, idx = 4*x;
            while(r < height) {
                uint8_t rgb[4] = {pixels[idx], pixels[idx + 1], pixels[idx + 2], 255};
                file.write((const char*) rgb, 4);
                r++; idx += 4*width;
            }
            order << 4*r << std::endl;
        }
    }

    bool binaryToImage(const std::string& binary, const std::string& order) {

        std::stringstream ss(order);
        char c; int width = 0, height = 0;
        std::string dim, remaining;
        bool orderGood = false;
        while(getline(ss, dim, '_')) {
            if(std::stringstream(dim) >> width >> c >> height) {
                if(c == 'x') {
                    orderGood = true;
                    break;
                }
            }
        }
        if(!orderGood) {
            std::cout << "ERROR: Image dimensions not found" << std::endl;
            return false;
        }

        sf::Texture texture;
        texture.create(width, height);
        sf::Uint8 pixels[width*height*4];

        std::ifstream bis(binary, std::ifstream::binary), ord(order);

        if(!bis.is_open()) {
            std::cout << "ERROR: Binary file not open!" << std::endl;
            return false;
        } else if(!ord.is_open()) {
            std::cout << "ERROR: Order file not open!" << std::endl;
            return false;
        } else {
            std::string line;
            while(std::getline(ord, line)) {
                std::stringstream ss(line);
                int x; ss >> x;

                uint8_t rgb[4];
                int r = 0;
                while(r < height) {
                    bis.read((char*) rgb, 4);
                    int idx = 4*(r * width + x);
                    pixels[idx] = rgb[0];
                    pixels[idx + 1] = rgb[1];
                    pixels[idx + 2] = rgb[2];
                    pixels[idx + 3] = rgb[3];
                    r++;
                }
            }

            std::string filename(binary);
            while(filename.back() != '.' && filename.length() > 1) {
                filename.pop_back();
            }
            filename.pop_back();

            texture.update(pixels);
            sf::Image image = texture.copyToImage();
            if(!image.saveToFile(filename + ".png")) {
                std::cout << "ERROR: Could not save to file" << std::endl;
                return false;
            }
            return true;
        }
    }

    void fullImageToBinary(sf::Uint8* pixels, int width, int height) {
        std::string path(getRawDate() + "_ORDERED_" + getImageDimensions(width, height) + ".bin");

        std::ofstream file(path, std::ofstream::binary | std::ofstream::app);
        if(!file.is_open()) {
            std::cout << "ERROR: Binary file not open!" << std::endl;
        } else {
            for(int x = 0; x < width; x++) {
                int idx = 4*x;
                for(int y = 0; y < height; y++) {
                    uint8_t rgb[4] = {pixels[idx], pixels[idx + 1], pixels[idx + 2], 255};
                    file.write((const char*) rgb, 4);
                    idx += 4*width;
                }
            }
        }
    }

    bool fullBinaryToImage(const std::string& binary) {
        std::stringstream ss(binary);
        std::string part;
        char c; int width, height; bool dimFound = false;
        while(getline(ss, part, '_')) {
            if(std::stringstream(part) >> width >> c >> height) {
                if(c == 'x') {
                    dimFound = true;
                    break;
                }
            }
        }

        if(!dimFound) {
            std::cout << "ERROR: Could not find file dimensions!" << std::endl;
        }

        sf::Texture texture;
        texture.create(width, height);
        sf::Uint8 pixels[width*height*4];

        std::ifstream bis(binary, std::ifstream::binary);

        if(!bis.is_open()) {
            std::cout << "ERROR: Binary file not open!" << std::endl;
            return false;
        } else {
            for(int x = 0; x < width; x++) {
                for(int y = 0; y < height; y++) {
                    uint8_t rgb[4];
                    bis.read((char*) rgb, 4);
                    int idx = 4*(x + y*width);
                    pixels[idx] = rgb[0];
                    pixels[idx + 1] = rgb[1];
                    pixels[idx + 2] = rgb[2];
                    pixels[idx + 3] = rgb[3];
                }
            }

            std::string filename(binary);
            while(filename.back() != '.' && filename.length() > 1) {
                filename.pop_back();
            }
            filename.pop_back();

            texture.update(pixels);
            sf::Image image = texture.copyToImage();
            if(!image.saveToFile(filename + ".png")) {
                std::cout << "ERROR: Could not save to file" << std::endl;
                return false;
            }
            return true;
        }
    }

    void halfImageToBinary(const std::string& path, sf::Uint8* pixels, int width, int height, bool left) {
        if(left) {
            for(int i = 0; i < width/2; i++) {
                imageToBinary(path, pixels, i, width, height);
            }
        } else {
            for(int i = width/height; i < width; i++) {
                imageToBinary(path, pixels, i, width, height);
            }
        }
    }

    void quarterImageToBinary(const std::string& path, sf::Uint8* pixels, int width, int height, int quarter) {
        quarter = quarter % 4;
        switch(quarter) {
            case 0:
                for(int i = 0; i < width/4; i++) {
                    imageToBinary(path, pixels, i, width, height);
                }
                return;
            case 1:
                for(int i = width/4; i < width/2; i++) {
                    imageToBinary(path, pixels, i, width, height);
                }
                return;
            case 2:
                for(int i = width/2; i < 3*width/4; i++) {
                    imageToBinary(path, pixels, i, width, height);
                }
                return;
            case 3:
                for(int i = 3*width/4; i < width; i++) {
                    imageToBinary(path, pixels, i, width, height);
                }
                return;
        }
    }

}

#endif