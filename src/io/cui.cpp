#include "cui.hpp"

bool cui::parseCommandLineInput(int argc, char *argv[], inputs& cmd) {
    std::vector<std::string> cmdInput;
    for(int i = 1; i < argc; i++) cmdInput.push_back(argv[i]);
    for(int i = 0; i < argc-1; i++) {
        if(cmdInput[i] == "?" || cmdInput[i] == "help") {
            std::cout << termcolor::yellow << "Program command line input:\n" << termcolor::reset << std::endl;
            std::cout <<    "main [-rt] [-f <file>] [-n, -N <int>] [-m <mode>] [?,help]\n"
                            "Explanation:\n"
                            " -rt: Launch into real-time interactive rendering mode\n"
                            " -f: Read configuration file with path <file> relative to root directory\n"
                            " -n,-N: Set number of rays per pixel to a positive integer (overrides value in <file>)\n"
                            " -m: Select static rendering mode, three options [single,s|batched,b|pixel,p]\n"
                            " ?,help: This screen\n"
            << std::endl;
            return false;
        }
        if(i+1 < argc-1) {
            if(cmdInput[i] == "-n" || cmdInput[i] == "-N") {
                std::cout << "parseInt: " << cmdInput[i+1] << std::endl;
                if(!readMethods::parseInt(cmdInput[++i], cmd.samples) || cmd.samples < 1) throw std::runtime_error("Samplecount must be a positive integer");
            }
            if(cmdInput[i] == "-f") {
                cmd.file = cmdInput[++i];
                if(!fs::exists(cmd.file)) throw std::runtime_error("Could not find file \"" + cmdInput[i] + "\"");
            }
            if(cmdInput[i] == "-m") {
                std::string mode = cmdInput[++i];
                aux::uppercase(mode);
                if(mode == "SINGLE" || mode == "S") {
                    cmd.launchMode = RenderMode::Single_full;
                } else if(mode == "BATCHED" || mode == "B") {
                    cmd.launchMode = RenderMode::Partial_full;
                } else if(mode == "PIXEL" || mode == "P") {
                    cmd.launchMode = RenderMode::Partial_pixel;
                } else {
                    throw std::runtime_error("Could not parse render mode");
                }
            }
        }
        if(cmdInput[i] == "-rt") cmd.realtime = true;
    }
    return true;
}

void cui::checkScene(inputs& cmd) {
    if(cmd.file.empty()) {
        std::cout << termcolor::yellow << 
            "No file provided. Run program with \"?\" for help. Currently selected scene marked with *.\n" 
            "Select another scene by providing a number, write a config file path, or press enter to continue" << std::endl;
        std::string line;
        while(true) {
            int sceneCount = 5;
            std::cout << termcolor::yellow << "\nScenes:" << termcolor::reset << std::endl;
            for(int i = 0; i < sceneCount; i++) {
                sceneType s = sceneType(i);
                if(s == cmd.scene) {
                    std::cout << i << ": " << termcolor::green << s << " *" << termcolor::reset << std::endl;
                } else {
                    std::cout << i << ": " << s << std::endl;
                }
            }
            std::getline(std::cin, line);
            if(line.size() > 0) {
                int val = -1;
                if(!readMethods::parseInt(line, val)) {
                    if(fs::exists(line)) {
                        cmd.file = line; break;
                    }
                    std::cout << termcolor::red << "Input was not int or a file was not found" << termcolor::reset << std::endl;
                    continue;
                }
                if(val < 0 || val > sceneCount-1) {
                    std::cout << termcolor::red << "Scene index out of bounds" << termcolor::reset << std::endl;
                    continue;
                }
                cmd.scene = sceneType(val);
                continue;
            }
            break;
        }
    }
}

void cui::overrideConfig(inputs& cmd, Config& conf) {
    if(cmd.scene != sceneType::EMPTY) {
        conf.scene = cmd.scene;
        if(cmd.scene == sceneType::PLATON) conf.background = backgroundType::NIGHT;
    }
}

void cui::printStart(const Config& conf) {
    std::cout << termcolor::yellow <<
    "#################################\n"
    "#        Ray tracer (GPU)       #\n"
    "# Date: " << aux::getDate() << " #\n"
    "#################################" 
    << termcolor::reset << std::endl;

    std::cout << "Resolution: " << conf.cam.width << "x" << conf.cam.height << ", N = " << conf.cam.samples << ", bounces = " << conf.cam.depth << std::endl;
    std::cout << "Backup to file: ";
    if(conf.backup) {
        std::cout << termcolor::bright_green;
    } else {
        std::cout << termcolor::bright_red;
    }
    std::cout << std::boolalpha << conf.backup << termcolor::reset << std::endl;
}