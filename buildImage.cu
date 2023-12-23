#include "imageBackupf.hpp"
#include "filesystem"

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

    // nvcc buildImage.cu -o buildImage -lsfml-graphics -lsfml-window -lsfml-system

    /**
     * @brief Builds the image from a backup file 
     * 
     */

    std::string current = ".";

    std::cout << "\033[0;93m##################################\033[0m" << std::endl;
    std::cout << "\033[0;93m#   Build an image from a file   #\033[0m" << std::endl;
    std::cout << "\033[0;93m##################################\033[0m" << std::endl;

    std::cout << "\033[0;93mFollowing backup files found:\033[0m" << std::endl;

    std::vector<std::string> files;

    int i = 0;
    for(auto f : std::filesystem::directory_iterator(current)) {
        if(f.is_regular_file() && f.path().extension() == ".txt" && !isColumnorder(f.path().stem().string())) {
            std::cout << "\033[1;34m" << i << "\033[0m" << ": " << f.path().filename() << std::endl;
            files.push_back(f.path().filename());
            i++;
        } else if(f.is_regular_file() && f.path().extension() == ".bin") {
            std::string column(current + "/" + f.path().stem().string() + "_columnorder.txt");
            if(std::filesystem::exists(std::filesystem::path(column)) || isOrderedBinary(f.path().stem().string())) { // Print only those that have a columnorder file OR are ordered
                std::cout << "\033[1;34m" << i << "\033[0m" << ": " << f.path().filename() << std::endl;
                files.push_back(f.path().filename());
                i++;
            }
        }
    }

    int idx = 0; std::string in;
    std::cout << "\033[0;93mSelect the file by giving its index:\033[0m ";
    getline(std::cin, in);
    std::stringstream(in) >> idx;
    while(idx >= files.size() || idx < 0) {
        std::cout << "\033[31mERROR: Index not found\033[0m" << std::endl;
        std::cout << "Try again: ";
        getline(std::cin, in);
        std::stringstream(in) >> idx;
    }
    std::cout << "Selected file: " << files[idx] << std::endl;

    if(files[idx].back() == 't') { // .txt-files
        int width = 1920, height = 1080;
        std::cout << "\033[0;93mProvide resolution or press enter to use 1920x1080:\033[0m ";
        std::string res;
        getline(std::cin, res);
        if(!res.empty()) {
            int w, h;
            std::stringstream(res) >> w >> h;
            width = (w > 0 ? w : width); height = (h > 0 ? h : height);
        }

        if(!Backup::textToImage(files[idx], width, height)) {
            std::cout << "\033[31mError: Could not save image\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;32mImage generated!\033[0m" << std::endl;
        }
    } else { // .bin files
        std::string filename(files[idx]);
        while(filename.back() != '.' && filename.length() > 1) {
            filename.pop_back();
        }
        filename.pop_back();

        if(isOrderedBinary(files[idx])) {
            if(!Backup::fullBinaryToImage(files[idx])) {
                std::cout << "\033[31mError: Could not save image\033[0m" << std::endl;
            } else {
                std::cout << "\033[1;32mImage generated!\033[0m" << std::endl;
            }
        } else {
            if(!Backup::binaryToImage(files[idx], filename + "_columnorder.txt")) {
                std::cout << "\033[31mError: Could not save image\033[0m" << std::endl;
            } else {
                std::cout << "\033[1;32mImage generated!\033[0m" << std::endl;
            }
        }
    }
    
    
    

    return 0;
}