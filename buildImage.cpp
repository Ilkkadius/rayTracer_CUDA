#include "imageBackupf.hpp"
#include "filesystem"
#include "termcolor.hpp"


int main() { 

    // g++ buildImage.cpp src/imageBackupf.cpp src/logMethods.cpp -o buildImage -Isrc -std=c++17 -lsfml-graphics -lsfml-window -lsfml-system
    // g++ buildImage.cpp src/imageBackupf.cpp src/logMethods.cpp -o buildImage -std=c++17 -w -Isrc -Idependencies/include -Ldependencies/lib -lsfml-graphics -lsfml-window -lsfml-system

    /**
     * @brief Builds the image from a backup file 
     * 
     */

    std::string current = ".";

    std::cout << termcolor::bright_yellow << 
    "##################################\n"
    "#   Build an image from a file   #\n"
    "##################################\n"
    "Following backup files found:" << termcolor::reset << std::endl;

    std::vector<std::string> files;

    int i = 0;
    for(auto f : std::filesystem::directory_iterator(current)) {
        if(f.is_regular_file() && f.path().extension() == ".bin") {
            std::cout << termcolor::bold << termcolor::blue << i << termcolor::reset << ": " << f.path().filename() << std::endl;
            files.push_back(f.path().filename().string());
            i++;
        }
    }

    int idx = 0; std::string in;
    std::cout << termcolor::yellow << "Select the file by giving its index:" << termcolor::reset;
    getline(std::cin, in);
    std::stringstream(in) >> idx;
    while(idx >= files.size() || idx < 0) {
        std::cout << termcolor::red << "ERROR: Index not found" << termcolor::reset << std::endl;
        std::cout << "Try again: ";
        getline(std::cin, in);
        std::stringstream(in) >> idx;
    }
    std::cout << "Selected file: " << files[idx] << std::endl;

    
    if(!Backup::binaryToImage(files[idx])) {
        std::cout << termcolor::red << "Error: Could not save image" << termcolor::reset << std::endl;
    } else {
        std::cout << termcolor::bold << termcolor::green << "Image generated!" << termcolor::reset << std::endl;
    }

    return 0;
}