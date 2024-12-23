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
typedef unsigned int uint;


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
        __host__ bool readVertex(std::string& str, dynVec<Vector3D>& vertices) {
            std::string v; double X, Y, Z;
            std::stringstream(str) >> v >> X >> Y >> Z;
            if(v != "v") {
                return false;
            }
            vertices.push_back(Vector3D(X,Y,Z));
            return true;
        }

        /**
         * @brief Given a line "str" containing UV data, the function reads it to "uv"
         * 
         * @param str 
         * @param uv 
         * @return true 
         * @return false 
         */
        __host__ bool readUV(std::string& str, dynVec<double*>& uv) {
            std::string vt; double U, V;
            std::stringstream(str) >> vt >> U >> V;
            if(vt != "vt") {
                return false;
            }
            double* d = new double[2]; d[0] = U; d[1] = V;
            uv.push_back(d);
            return true;
        }

        /**
         * @brief Given a line "str" containing normal vector data, the function reads it to "normals"
         * 
         * @param str 
         * @param normals 
         * @return true 
         * @return false 
         */
        __host__ bool readNormal(std::string& str, dynVec<Vector3D>& normals) {
            std::string vn; double X, Y, Z;
            std::stringstream(str) >> vn >> X >> Y >> Z;
            if(vn != "vn") {
                return false;
            }
            normals.push_back(Vector3D(X,Y,Z));
            return true;
        }

        /**
         * @brief Return the wanted index from a face element, which must be of the form: f/uv/vn
         * 
         * @param faceElem f/uv/vn
         * @param elemIdx index: 0, 1 or 2
         * @return int, value of f, uv or vn
         */
        __host__ int FaceElemIdx(const std::string& faceElem, int elemIdx) {
            std::stringstream ss(faceElem);
            int index; std::string buffer;
            for(int i = 0; i < elemIdx + 1; i++)
                std::getline(ss, buffer, '/');
            if(std::stringstream(buffer) >> index)
                return index;
            return -1;
        }

        /**
         * @brief Separates the face definition part from a possible color definition part
         * 
         * @param line          e.g. f 1/2/3 4/5/6 7/8/9 # 0.1 0.2 0.9
         * @param faceData      f 1/2/3 4/5/6 7/8/9
         * @param faceColor     Vector(0.1, 0.2, 0.9)
         * @return true         if successfull,
         * @return false        otherwise
         */
        __host__ bool FaceSepColor(std::string& line, std::string& faceData, Vector3D& faceColor) {
            if(std::getline(std::stringstream(line), faceData, '#')) {
                if(line.length() > faceData.length()) {
                    double r, g, b;
                    faceColor = (std::stringstream(line.substr(faceData.length() + 1)) >> r >> g >> b) ? Vector3D(r, g, b) : faceColor;
                    std::cout << faceColor << std::endl;
                }
                return true;
            }
            return false;
        }

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
        __host__ bool readTriangle(std::stringstream& ss, std::string& firstVertex, std::string& secondVertex, dynVec<int*>& fVertices, dynVec<int*>& fNormals) {
            std::string v;
            if(!(ss >> v)) return false;

            int v0 = FaceElemIdx(firstVertex, 0);
            int v1 = FaceElemIdx(secondVertex, 0);
            int v2 = FaceElemIdx(v, 0);
            int* vs = new int[3]; vs[0] = v0 - 1; vs[1] = v1 - 1; vs[2] = v2 - 1; // CORRECT INDEXING
            fVertices.push_back(vs);

            int n0 = FaceElemIdx(firstVertex, 2);
            int n1 = FaceElemIdx(secondVertex, 2);
            int n2 = FaceElemIdx(v, 2);
            int* ns = new int[3]; ns[0] = n0 - 1; ns[1] = n1 - 1; ns[2] = n2 - 1; // CORRECT INDEXING
            (n0 > 0 && n1 > 0 && n2 > 0) ? fNormals.push_back(ns) : fNormals.push_back(NULL);

            secondVertex = v;
            
            return true;
        }


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
        __host__ bool readFace(std::string& line, dynVec<int*>& fVertices, dynVec<Vector3D>& fColors, dynVec<int*>& fNormals) {
            std::string faceData;
            Vector3D faceColor(-1.0,-1.0,-1.0);

            if(!FaceSepColor(line, faceData, faceColor)) // Separate face data and get the (possible) color of the face
                return false;
                
            std::string f, first, second;
            std::stringstream ss(faceData);
            
            if( !(ss >> f >> first >> second) && (f != "f") ) return false;
            while(readTriangle(ss, first, second, fVertices, fNormals))
                fColors.push_back(faceColor);
            return true;
        }

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
            dynVec<Vector3D>& normals, dynVec<int*>& fVertices, dynVec<int*>& fNormals, dynVec<Vector3D>& fColors) {
            std::string line;
            bool flag = false;
            if(is.is_open()) {
                while(std::getline(is, line)) {
                    if(line[0] == 'v') {
                        std::string str = line.substr(0,2);
                        if(str == "v ") {
                            ReadFunctions::readVertex(line, vertices);
                        } else if(str == "vt") {
                            //ReadFunctions::readUV(line, uv);
                        } else if(str == "vn") {
                            ReadFunctions::readNormal(line, normals);
                        }
                    } else if(line[0] == 'f') {
                        flag = true;
                        ReadFunctions::readFace(line, fVertices, fColors, fNormals);
                    } else {
                        if(flag) { // If a vertex group has been read completely, i.e. f lines have ended, break from the loop
                            break;
                        }
                    }
                }
                if(!is.good()) {
                    is.close();
                    return false;
                }
                return true;
            }
            return false;
        }
        

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
    __host__ bool TargetsFromFile(const char* path, targetList** list, Shape** shapes, const Vector3D& defaultColor = Vector3D(0.4,0.5,1)) {
        dynVec<Vector3D> vertices(1<<7);                        // Store all vertices "v"
        dynVec<Vector3D> fColors(1<<7);                         // Store colors "#"
        dynVec<double*> uv(1<<7);                               // Store uv data "vt"
        dynVec<Vector3D> normals(1<<7);                         // Store all normals "vn"
        dynVec<int*> fVertices(1<<7), fNormals(1<<7);           // Save the indices given by face elements f/vt/fn

        std::ifstream is(path);
        if(!is.is_open()) {
            std::cout << "Error in opening the file to read targets" << std::endl;
            return false;
        }
        
        while(ReadFunctions::ReadGroup(is, vertices, uv, normals, fVertices, fNormals, fColors)) {}
        
        Vector3D* verts, * fCols; int* fVerts;
        CHECK(cudaMallocManaged(&verts, vertices.size() * sizeof(Vector3D)));
        CHECK(cudaMallocManaged(&fCols, fColors.size() * sizeof(Vector3D)));
        CHECK(cudaMallocManaged(&fVerts, 3 * fVertices.size() * sizeof(int*)));
        CHECK(cudaDeviceSynchronize());
        
        for(int i = 0; i < vertices.size(); i++) {
            verts[i] = vertices[i];
        }
        for(int i = 0; i < fVertices.size(); i++) {
            int idx = 3*i; int* fv = fVertices[i];
            fVerts[idx] = fv[0];
            fVerts[idx + 1] = fv[1];
            fVerts[idx + 2] = fv[2];
        }

        for(int i = 0; i < fColors.size(); i++) {
            fCols[i] = fColors[i];
        }

        Vector3D* defCol;
        CHECK(cudaMallocManaged(&defCol, sizeof(Vector3D)));
        *defCol = defaultColor;

        generateTargets<<<1, 1>>>(list, shapes, verts, fVerts, fCols, fVertices.size(), defCol);
        CHECK(cudaDeviceSynchronize());

        if(is.is_open()) {
            is.close();
        }

        for(int i = 0; i < uv.size(); i++) {
            delete[] uv[i];
        }
        for(int i = 0; i < fVertices.size(); i++) {
            delete[] fVertices[i];
        }
        for(int i = 0; i < fNormals.size(); i++) {
            delete[] fNormals[i];
        }

        CHECK(cudaFree(verts));
        CHECK(cudaFree(fCols));
        CHECK(cudaFree(fVerts));

        return true;
    }

    /**
     * @brief Generate compounds from .obj file given by "path". NB: Currently generates only one compound
     * 
     * @param path 
     * @param list 
     * @param listCount 
     * @param defaultColor 
     * @return __host__ 
     */
    __host__ bool CompoundsFromFile(const char* path, Compound** list, size_t& listSize, const Vector3D& defaultColor = Vector3D(0.4,0.5,1)) {
        dynVec<Vector3D> vertices(1<<7);                        // Store all vertices "v"
        dynVec<Vector3D> fColors(1<<7);                         // Store colors "#"
        dynVec<double*> uv(1<<7);                               // Store uv data "vt"
        dynVec<Vector3D> normals(1<<7);                         // Store all normals "vn"
        dynVec<int*> fVertices(1<<7), fNormals(1<<7);           // Save the indices given by face elements f/vt/fn

        std::ifstream is(path);
        if(!is.is_open()) {
            std::cout << "Error in opening the file to read targets" << std::endl;
            return false;
        }
        
        while(ReadFunctions::ReadGroup(is, vertices, uv, normals, fVertices, fNormals, fColors)) {}
        
        Vector3D* verts, * fCols; int* fVerts;
        CHECK(cudaMallocManaged(&verts, vertices.size() * sizeof(Vector3D)));
        CHECK(cudaMallocManaged(&fCols, fColors.size() * sizeof(Vector3D)));
        CHECK(cudaMallocManaged(&fVerts, 3 * fVertices.size() * sizeof(int*)));
        CHECK(cudaDeviceSynchronize());
        
        for(int i = 0; i < vertices.size(); i++) {
            verts[i] = vertices[i];
        }
        for(int i = 0; i < fVertices.size(); i++) {
            int idx = 3*i; int* fv = fVertices[i];
            fVerts[idx] = fv[0];
            fVerts[idx + 1] = fv[1];
            fVerts[idx + 2] = fv[2];
        }

        for(int i = 0; i < fColors.size(); i++) {
            fCols[i] = fColors[i];
        }

        Vector3D* defCol;
        CHECK(cudaMallocManaged(&defCol, sizeof(Vector3D)));
        *defCol = defaultColor;

        generateCompounds<<<1, 1>>>(list, verts, fVerts, fCols, fVertices.size(), defCol); listSize = 1;
        CHECK(cudaDeviceSynchronize());

        if(is.is_open()) {
            is.close();
        }

        for(int i = 0; i < uv.size(); i++) {
            delete[] uv[i];
        }
        for(int i = 0; i < fVertices.size(); i++) {
            delete[] fVertices[i];
        }
        for(int i = 0; i < fNormals.size(); i++) {
            delete[] fNormals[i];
        }

        CHECK(cudaFree(verts));
        CHECK(cudaFree(fCols));
        CHECK(cudaFree(fVerts));

        return true;
    }


} // End fileOperations namespace

#endif