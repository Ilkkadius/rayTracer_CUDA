#ifndef SCENE_CONFIG_FILE_READ_CUDA_HPP
#define SCENE_CONFIG_FILE_READ_CUDA_HPP

#include "target.hpp"
#include "cameraf.hpp"
#include "renderMode.hpp"

#include "readMethods.hpp"
#include "meshRead.hpp"
#include "csgRead.hpp"

using namespace readMethods;

namespace fileRead {


    __host__ void parseFile(const char* path, TargetList** list, Config& conf);

    __host__ void parseConfig(streamHolder sh, std::string& line, uint& linenum, Config& conf);
    __host__ void parseScene(streamHolder sh, std::string& line, uint& linenum, TargetList** list);
    __host__ void parsePrimitives(streamHolder sh, std::string& line, uint& linenum, TargetList** list);

    __global__ void generateTargets(TargetList** list, targetData* data, affine* affines, int targetCount);

    __host__ void getShapeTransformation(streamHolder sh, std::string& line, uint& linenum, affine& aff);

    __host__ void skipLine(streamHolder sh, std::string& line, uint& linenum);
};

#endif