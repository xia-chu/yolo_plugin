cmake_minimum_required(VERSION 3.5)

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(PROJECT_NAME Yolov8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

# -------------- ONNXRUNTIME  ------------------#
if (NOT DEFINED ONNXRUNTIME_ROOT OR ONNXRUNTIME_ROOT STREQUAL "")
    message(FATAL_ERROR "ONNXRUNTIME_ROOT is required")
endif ()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)

# 添加插件
add_subdirectory(plugin)

set(PROJECT_SOURCES
        main.cpp
        inference.h
        inference.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
endif ()

# For windows system, copy onnxruntime.dll to the same folder of the executable file
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()

# Download https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml
# and put it in the same folder of the executable file
configure_file(coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)

# Copy yolov8n.onnx file to the same folder of the executable file
configure_file(yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx COPYONLY)

# Create folder name images in the same folder of the executable file
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)
