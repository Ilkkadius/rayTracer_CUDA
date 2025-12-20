cmake_minimum_required(VERSION 3.24)

project(main LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

message(CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID})
message(CUDA_ARCITECTURES=${CMAKE_CUDA_ARCHITECTURES})
message(CUDA_FLAGS=${CMAKE_CUDA_FLAGS})

#set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)
message(SFML_FOUND=${SFML_FOUND})
message(SFML_DIR=${SFML_DIR})

include_directories(src ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDASOURCES "src/*.cu")

target_link_libraries(main sfml-graphics)
target_sources(main PRIVATE ${SOURCES} ${CUDASOURCES})