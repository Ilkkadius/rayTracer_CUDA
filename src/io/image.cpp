#include "image.hpp"

void Image::ToBinary(sf::Uint8* pixels, int width, int height) {
    std::lock_guard<std::mutex> lock(binaryMutex);
    std::string path(getRawDate() + "_" + getImageDimensions(width, height) + ".bin");

    std::ofstream file(path, std::ofstream::binary);
    if(!file.is_open()) {
        std::cout << "ERROR: Binary file not open!" << std::endl;
    } else {
        file.write((char*)&width, sizeof(int));
        file.write((char*)&height, sizeof(int));
        file.write((char*)pixels, 4*width*height*sizeof(sf::Uint8)); // Memory efficiency by discarding fourth value?
    }
    file.close();
}

bool Image::fromBinary(const std::string& path) {
    std::ifstream file(path, std::ifstream::binary);
    
    if(!file.is_open()) {
        std::cout << "ERROR: Binary file not open!" << std::endl;
        return false;
    }

    int width, height;
    file.read((char*)&width,sizeof(int));
    file.read((char*)&height,sizeof(int));
    
    if(width > 1920 && height > 1080) std::cout << "WARNING: Image dimensions large: " << width << "x" << height << std::endl;

    sf::Uint8* pixels = new sf::Uint8[4*width*height];

    file.read((char*)pixels, 4*width*height*sizeof(sf::Uint8));
    file.close();

    sf::Texture texture;
    texture.create(width, height);
    texture.update(pixels);
    delete[] pixels;

    std::string filename = path.substr(0, path.size() - 4);

    sf::Image image = texture.copyToImage();
    if(!image.saveToFile(filename + ".png")) {
        return false;
    }
    return true;
}

void Image::writePNG(sf::Uint8* pixels, int width, int height, int samples, double duration) {
    sf::Texture texture;
    texture.create(width, height);
    texture.update(pixels);

    std::string filename = getImageFilename(width, height, samples, duration);

    sf::Image image = texture.copyToImage();

    std::string prefix = "figures/";
    if(!image.saveToFile(prefix + filename)) {
        std::cout << "Trying again..." << std::endl;
        prefix = "../" + prefix;
        if(!image.saveToFile(prefix + filename)) {
            std::cout << "Trying again..." << std::endl;
            if(!image.saveToFile(filename)) {
                std::cout << termcolor::red << "Image was not saved..." << termcolor::reset << std::endl;
            } else {
                std::cout << "Successfully saved the image to the current directory" << std::endl;
            }

        } else {
            std::cout << "Successfully saved the image \"" << prefix + filename << "\"" << std::endl;
        }    
    } else {
        std::cout << "Successfully saved the image \"" << prefix + filename << "\"" << std::endl;
    }
}

