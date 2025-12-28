#ifndef READ_METHODS_CUDA_HPP
#define READ_METHODS_CUDA_HPP

#include <sstream>
#include <fstream>

#include "target.hpp"
#include "cameraf.hpp"
#include "backgroundsf.hpp"
#include "renderMode.hpp"

#define RENDER_CONFIG \
    X(sceneType, scene, sceneType::EMPTY) \
    X(Camera, cam, Camera()) \
    X(backgroundType, background, backgroundType::DARKNESS) \
    X(RenderMode, mode, RenderMode::Single_full) \
    X(bool, backup, false) \
    X(bool, realtime, false)

typedef struct {

    #define X(a,b,c) a b {c};
    RENDER_CONFIG
    #undef X

} Config;

namespace readMethods {
    typedef struct {
        const char* path;
        std::ifstream& file;
        std::stringstream& ss;
    } streamHolder;

    typedef struct {
        targetType type;
        float params[13];
    } targetData;


    typedef struct {
        Matrix A;
        Vector3D b;
    } affine;

    __host__ inline bool readVector3D(std::stringstream& ss, Vector3D& vec) { // TODO: Able to handle multiple lines
        return (ss >> vec.x) && (ss >> vec.y) && (ss >> vec.z);
    }
    __host__ inline bool findNext(streamHolder sh, std::string& line, uint& linenum)  {
        if(sh.ss >> line) {
            return true;
        } else {
            while(std::getline(sh.file, line)) {
                linenum++; sh.ss.str(line); sh.ss.clear();
                if(sh.ss >> line) return true;
            }
        }
        return false;
    }

    __host__ inline bool parseFloat(const std::string& line, float& val) {
        try {
            val = std::stof(line);
            return true;
        } catch (...) {
            return false;
        }
    }
    __host__ inline bool parseInt(const std::string& line, int& val) {
        try {
            val = std::stoi(line);
            return true;
        } catch (...) {
            return false;
        }
    }
    __host__ inline bool parseBool(std::string line, bool& val) {
        aux::uppercase(line);
        if(line == "TRUE") {
            val = true;
            return true;
        } else if(line == "FALSE") {
            val = false;
            return true;
        } else {
            return false;
        }
    }

    __host__ inline void readTransformation(streamHolder sh, std::string& line, uint& linenum, affine& aff) {
        findNext(sh, line, linenum);
        if(line != "{") aux::error(sh.path, linenum, "Transformation: Expected \"{\", found \"" + line + "\".");
        bool ready = false;
        while(findNext(sh, line, linenum)) {
            aux::uppercase(line);
            if(line == "MOVE") {
                Vector3D b;
                if(!readVector3D(sh.ss, b)) aux::error(sh.path, linenum, "Transformation: Could not read translation vector.");
                aff.b += b; // or aff.A * b;
            } else if(line == "ROTATE") {
                Vector3D axis, pos; float angle;
                if(!readVector3D(sh.ss, axis)) aux::error(sh.path, linenum, "Transformation: Could not read axis.");
                if(!(sh.ss >> angle)) aux::error(sh.path, linenum, "Transformation: Could not read angle.");
                if(!readVector3D(sh.ss, pos)) aux::error(sh.path, linenum, "Transformation: Could not read axis position.");
                Matrix rot = generateRotation(angle * M_PI/180.0f, unitVec(axis));
                aff.A = aff.A * rot;
            } else if(line == "}") {
                ready = true;
                break;
            }
        }

        if(!ready) aux::error(sh.path, linenum, "Transformation: Definition does not end, forgotten \"}\"?");
    }
};



#endif