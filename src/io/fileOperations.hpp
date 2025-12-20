#ifndef CUDA_FILE_METHODS_TO_WRITE_OR_READ_TARGETS_HPP
#define CUDA_FILE_METHODS_TO_WRITE_OR_READ_TARGETS_HPP

#include <cuda_runtime.h>

#include <vector>
#include <sstream>
#include <fstream>

#include "vector3D.hpp"
#include "targetf.hpp"
#include "compoundf.hpp"
#include "kernelSet.hpp"

// NB: .obj file indexing starts from 1

namespace FileOperations{

    /**
     * @brief Auxiliary functions not to be used outside FileOperations
     * 
     */
    namespace ReadFunctions {
        /**
         * @brief Given a line "str" containing a definition of a vertex, the function reads it to "vertices"
         * 
         * @param str 
         * @param vertices 
         * @return true, if successfull
         * @return false 
         */
        __host__ bool readVertex(std::string& str, dynVec<Vector3D>& vertices);

        /**
         * @brief Given a line "str" containing UV data, the function reads it to "uv"
         * 
         * @param str 
         * @param uv 
         * @return true 
         * @return false 
         */
        __host__ bool readUV(std::string& str, dynVec<double*>& uv);

        /**
         * @brief Given a line "str" containing normal vector data, the function reads it to "normals"
         * 
         * @param str 
         * @param normals 
         * @return true 
         * @return false 
         */
        __host__ bool readNormal(std::string& str, dynVec<Vector3D>& normals);

        /**
         * @brief Return the wanted index from a face element, which must be of the form: f/uv/vn
         * 
         * @param faceElem f/uv/vn
         * @param elemIdx index: 0, 1 or 2
         * @return int, value of f, uv or vn
         */
        __host__ int FaceElemIdx(const std::string& faceElem, int elemIdx);

        /**
         * @brief Separates the face definition part from a possible color definition part
         * 
         * @param line          e.g. f 1/2/3 4/5/6 7/8/9 # 0.1 0.2 0.9
         * @param faceData      f 1/2/3 4/5/6 7/8/9
         * @param faceColor     Vector(0.1, 0.2, 0.9)
         * @return true         if successfull,
         * @return false        otherwise
         */
        __host__ bool FaceSepColor(std::string& line, std::string& faceData, Vector3D& faceColor);

        /**
         * @brief Read the next triangle from stringstream containing "f"-line
         * 
         * @param ss            e.g. 1/2/3 4/5/6 7/8/9
         * @param firstVertex   e.g. 1/2/3
         * @param secondVertex  e.g. 4/5/6
         * @param fVertices     e.g. {1, 4, 7}
         * @param fNormals      e.g. {3, 6, 9}
         * @return true         if successfull,
         * @return false        otherwise
         */
        __host__ bool readTriangle(std::stringstream& ss, std::string& firstVertex, 
                                    std::string& secondVertex, dynVec<int*>& fVertices, dynVec<int*>& fNormals);


        /**
         * @brief Given a string containing a declaration of a face, the function reads it and its color information, if provided
         * 
         * @param line          e.g. f 1/2/3 4/5/6 7/8/9 10/11/12 # 0.1 0.2 0.9
         * @param fVertices     {1, 4, 7}, {1, 7, 10}
         * @param fColors       Vector3D(0.1, 0.2, 0.9), Vector3D(0.1, 0.2, 0.9)
         * @param fNormals      {3, 6, 9}, {6, 9, 12}
         * @return true         if successfull,
         * @return false        otherwise
         */
        __host__ bool readFace(std::string& line, dynVec<int*>& fVertices, dynVec<Vector3D>& fColors, dynVec<int*>& fNormals);

        /**
         * @brief Reads one group from vertex definitions to face definitions
         * 
         * @param is        the file stream to be read
         * @param vertices  container for the vertice position vectors
         * @param uv        container for the uv data
         * @param normals   container for the normal vectors
         * @param fVertices container for vertex indices to generate faces
         * @param nIndices  container for the face normals
         * @param fColors   container for the colors of the face
         * @return true     true, if there is more to be read in the file
         * @return false    false, otherwise
         */
        __host__ bool ReadGroup(std::ifstream& is, dynVec<Vector3D>& vertices, dynVec<double*>& uv, 
            dynVec<Vector3D>& normals, dynVec<int*>& fVertices, dynVec<int*>& fNormals, dynVec<Vector3D>& fColors);
        

    } // end ReadFunctions


    // ##################################################################################

    /**
     * @brief Read target data from file
     * 
     * @param path          .obj file to be read
     * @param targets       container for the read targets
     * @param defaultColor  color to be given for targets without already defined color
     * @return true, if read was successfull
     * @return false, otherwise
     */
    __host__ bool TargetsFromFile(const char* path, TargetList** list, Shape** shapes, const Vector3D& defaultColor = Vector3D(0.4,0.5,1));

    /**
     * @brief Generate compounds from .obj file given by "path". NB: Currently generates only one compound
     * 
     * @param path 
     * @param list 
     * @param listCount 
     * @param defaultColor 
     * @return __host__ 
     */
    __host__ bool CompoundsFromFile(const char* path, Compound** list, size_t& listSize, const Vector3D& defaultColor = Vector3D(0.4,0.5,1));


} // End fileOperations namespace

#endif