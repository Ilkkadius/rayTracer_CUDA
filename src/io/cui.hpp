#ifndef COMMAND_LINE_UI_CUDA_HPP
#define COMMAND_LINE_UI_CUDA_HPP

#include <vector>
#include <filesystem>
#include "termcolor.hpp"

#include "readMethods.hpp"

namespace fs = std::filesystem;

namespace cui {

    typedef struct {
        std::string file = "";
        int samples = 0;
        RenderMode launchMode = RenderMode::Single_full;
        sceneType scene = sceneType::EMPTY;
        bool realtime = false;
    } inputs;

    /**
     * @brief Parse command line inputs when program starts
     * 
     * @param argc 
     * @param argv 
     * @param in Variable holder for definitions from command line
     * @return true if program can continue
     * @return false if program should return
     */
    bool parseCommandLineInput(int argc, char *argv[], inputs& cmd);

    /**
     * @brief Ask user to select scene, if configuration file was not provided
     * 
     * @param cmd 
     */
    void checkScene(inputs& cmd);

    /**
     * @brief Set the command line parameter priority over configuration file definitions
     * 
     * @param cmd 
     * @param conf 
     */
    void overrideConfig(inputs& cmd, Config& conf);

    void printStart(const Config& conf);

};

#endif