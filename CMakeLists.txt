#
# mnist

cmake_minimum_required(VERSION 3.17)

project("mnist.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ggml)

add_executable(mnist main.cpp)

target_include_directories(mnist PUBLIC .)
target_compile_features(mnist PUBLIC cxx_std_20)
target_link_libraries(mnist PRIVATE ggml)
