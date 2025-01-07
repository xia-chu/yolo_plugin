# -------------- FFmpeg  ------------------#
find_package(PkgConfig REQUIRED)
# 查找 ffmpeg/libutil 是否安装
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)

set(plugin_name "yolov8")
add_library(${plugin_name} SHARED "yolo_plugin.cpp" "../inference.cpp")
# 添加头文件目录
target_include_directories(${plugin_name} PRIVATE PkgConfig::AVUTIL)
# 链接库
target_link_libraries(${plugin_name} PkgConfig::AVUTIL)

if (MSVC)
    add_definitions(-DPLUGIN_EXPORTS)
endif ()

if (WIN32)
    target_link_libraries(${plugin_name} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${plugin_name} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${plugin_name} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${plugin_name} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${plugin_name} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
endif ()