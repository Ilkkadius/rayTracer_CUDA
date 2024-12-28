#include "imageBackupf.hpp"
#include "filesystem"
#include "termcolor.hpp"

__host__ bool isOrderedBinary(const std::string& stem) {
    std::stringstream ss(stem);
    std::string part;
    while(getline(ss, part, '_')) {
        if(part == "ORDERED") {
            return true;
        }
    }
    return false;
}

__host__ bool isColumnorder(const std::string& stem) {
    std::stringstream ss(stem);
    std::string part;
    while(getline(ss, part, '_')) {
        if(part == "columnorder") {
            return true;
        }
    }
    return false;
}

int main() { 

    // nvcc buildImage.cu -o buildImage -std=c++17 -lsfml-graphics -lsfml-window -lsfml-system
    // nvcc buildImage.cu src/imageBackupf.cpp src/logMethods.cpp -o buildImage -std=c++17 -w -Isrc -Idependencies/include -Ldependencies/lib -lsfml-graphics -lsfml-window -lsfml-system

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
        if(f.is_regular_file() && f.path().extension() == ".txt" && !isColumnorder(f.path().stem().string())) {
            std::cout << termcolor::bold << termcolor::blue << i << termcolor::reset << ": " << f.path().filename() << std::endl;
            files.push_back(f.path().filename().string());
            i++;
        } else if(f.is_regular_file() && f.path().extension() == ".bin") {
            std::string column(current + "/" + f.path().stem().string() + "_columnorder.txt");
            if(std::filesystem::exists(std::filesystem::path(column)) || isOrderedBinary(f.path().stem().string())) { // Print only those that have a columnorder file OR are ordered
                std::cout << termcolor::bold << termcolor::blue << i << termcolor::reset << ": " << f.path().filename() << std::endl;
                files.push_back(f.path().filename().string());
                i++;
            }
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

    if(files[idx].back() == 't') { // .txt-files
        int width = 1920, height = 1080;
        std::cout << termcolor::bright_yellow << "Provide resolution or press enter to use 1920x1080: " << termcolor::reset;
        std::string res;
        getline(std::cin, res);
        if(!res.empty()) {
            int w, h;
            std::stringstream(res) >> w >> h;
            width = (w > 0 ? w : width); height = (h > 0 ? h : height);
        }

        if(!Backup::textToImage(files[idx], width, height)) {
            std::cout << termcolor::red << "Error: Could not save image" << termcolor::reset << std::endl;
        } else {
            std::cout << termcolor::bold << termcolor::green << "Image generated!" << termcolor::reset << std::endl;
        }
    } else { // .bin files
        std::string filename(files[idx]);
        while(filename.back() != '.' && filename.length() > 1) {
            filename.pop_back();
        }
        filename.pop_back();

        if(isOrderedBinary(files[idx])) {
            if(!Backup::fullBinaryToImage(files[idx])) {
                std::cout << termcolor::red << "Error: Could not save image" << termcolor::reset << std::endl;
            } else {
                std::cout << termcolor::bold << termcolor::green << "Image generated!" << termcolor::reset << std::endl;
            }
        } else {
            if(!Backup::binaryToImage(files[idx], filename + "_columnorder.txt")) {
                std::cout << termcolor::red << "Error: Could not save image" << termcolor::reset << std::endl;
            } else {
                std::cout << termcolor::bold << termcolor::green << "Image generated!" << termcolor::reset << std::endl;
            }
        }
    }
    
    
    

    return 0;
}