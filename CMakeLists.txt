cmake_minimum_required(VERSION 3.15)
project(QuantEngine)

set(CMAKE_CXX_STANDARD 17)
# Rutas a la librería
include_directories(${CMAKE_SOURCE_DIR}/external/onnxruntime/include)
link_directories(${CMAKE_SOURCE_DIR}/external/onnxruntime/lib)
# Crear el ejecutable
add_executable(quant_engine src/engine_c/main.cpp)
# Lincar la librería dinámica de ONNX
target_link_libraries(quant_engine onnxruntime)