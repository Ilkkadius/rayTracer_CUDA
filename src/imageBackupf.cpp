#include "imageBackupf.hpp"

void Backup::imageToBinary(sf::Uint8* pixels, int width, int height) {
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

bool Backup::binaryToImage(const std::string& path) {
    std::ifstream file(path, std::ifstream::binary);
    
    if(!file.is_open()) {
        std::cout << "ERROR: Binary file not open!" << std::endl;
        return false;
    }

    int width, height;
    file.read((char*)&width,sizeof(int));
    file.read((char*)&height,sizeof(int));
    
    if(width > 1920 && height > 1080) std::cout << "Image dimensions large: " << width << "x" << height << std::endl;

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

