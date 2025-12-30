#ifndef CSG_READ_METHODS_CUDA_HPP
#define CSG_READ_METHODS_CUDA_HPP

#include <cuda_runtime.h>

#include <vector>
#include <sstream>
#include <fstream>

#include "fileRead.hpp"

using namespace readMethods;

namespace csgRead {

    __global__ void generateCSG(TargetList** list, uint* firstList, CSG* operList, int nodeCount, targetData* data, affine* affines, int targetCount);

    __host__ void parseCSG(streamHolder sh, std::string& line, uint& linenum, TargetList** list);

    __host__ void processLeft(streamHolder sh, int ptr, std::string& line, std::vector<uint>& firstList, std::vector<CSG>& operList, std::vector<affine>& affines, int& nodeCounter, int& targetCounter, uint& linenum, affine aff);
    __host__ void processRight(streamHolder sh, int ptr, std::string& line, std::vector<uint>& firstList, std::vector<CSG>& operList, std::vector<affine>& affines, int& nodeCounter, int& targetCounter, uint& linenum, affine aff);

    __host__ bool readTargets(const char* path, std::vector<targetData>& data, uint linenum);

    __host__ void skipTransformation(streamHolder sh, std::string& line, uint& linenum);

    __host__ void findStart(streamHolder sh, std::string& line, uint& linenum, affine& aff);
    __host__ void findEnd(streamHolder sh, std::string& line, uint& linenum);

};

#endif