//
// Created by xzl on 2025/1/4.
//

#include <cassert>
#include <iomanip>
#include <fstream>
#include <thread>
#include "video_plugin.h"
#include "../inference.h"

static const plugin_loader *s_loader = nullptr;
#define PLUGIN_LOG(lev, ...) s_loader->printf_log(lev, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
#define LOG_T(...) PLUGIN_LOG(0, ##__VA_ARGS__)
#define LOG_D(...) PLUGIN_LOG(1, ##__VA_ARGS__)
#define LOG_I(...) PLUGIN_LOG(2, ##__VA_ARGS__)
#define LOG_W(...) PLUGIN_LOG(3, ##__VA_ARGS__)
#define LOG_E(...) PLUGIN_LOG(4, ##__VA_ARGS__)


static std::vector<std::string> s_classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                                          "boat", "traffic light", "fire hydrant",
                                          "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                                          "cow", "elephant", "bear", "zebra",
                                          "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                          "skis", "snowboard", "sports ball", "kite",
                                          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                          "bottle", "wine glass", "cup", "fork", "knife",
                                          "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                                          "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                          "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                          "mouse", "remote", "keyboard", "cell phone",
                                          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                                          "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

struct plugin_instance {
    YOLO_V8 yoloDetector;

#if 0
    int ReadCocoYaml(const char *yaml) {
        // Open the YAML file
        std::ifstream file(yaml);
        if (!file.is_open()) {
            LOG_E("Failed to open file: %s", yaml);
            return 1;
        }

        // Read the file line by line
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }

        // Find the start and end of the names section
        std::size_t start = 0;
        std::size_t end = 0;
        for (std::size_t i = 0; i < lines.size(); i++) {
            if (lines[i].find("names:") != std::string::npos) {
                start = i + 1;
            } else if (start > 0 && lines[i].find(':') == std::string::npos) {
                end = i;
                break;
            }
        }

        // Extract the names
        std::vector<std::string> names;
        for (std::size_t i = start; i < end; i++) {
            std::stringstream ss(lines[i]);
            std::string name;
            std::getline(ss, name, ':'); // Extract the number before the delimiter
            std::getline(ss, name); // Extract the string after the delimiter
            names.push_back(std::move(name));
        }

        this->yoloDetector.classes = std::move(names);
        return 0;
    }
#endif

    int Detector(AVFrame *frame) {
        if (frame->format != AV_PIX_FMT_RGB24) {
            LOG_E("Input AVFrame is not in RGB24 format");
            return -1;
        }
        cv::Mat img(frame->height, frame->width, CV_8UC3, frame->data[0], frame->linesize[0]);
        std::vector<DL_RESULT> res;
        yoloDetector.RunSession(img, res);
        for (auto &re: res) {
            cv::RNG rng(cv::getTickCount());
            cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            cv::rectangle(img, re.box, color, 3);
            float confidence = floor(100 * re.confidence) / 100;
            std::string label = yoloDetector.classes[re.classId] + " " +
                                std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
            cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
            );

            cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
            );
        }
        return 0;
    }

};

static const char *s_plugin_name() {
    return "yolov8";
}

static void s_plugin_onload(const plugin_loader *loader) {
    assert(loader);
    s_loader = loader;
    LOG_I("plugin: %s loaded", s_plugin_name());
}

static void s_plugin_onunload() {
    LOG_I("start unload plugin: %s", s_plugin_name());
    s_loader = nullptr;
}

static int s_plugin_max_threads() {
    return std::thread::hardware_concurrency();
}

static int s_plugin_instance_create(plugin_instance **ptr, void *config_map, void *err) {
    assert(ptr && config_map);
    *ptr = (plugin_instance *) malloc(sizeof(plugin_instance));
    auto map = (std::map<std::string, std::string> *) config_map;
    auto e = (std::string *) err;

#if 0
    if ((*ptr)->ReadCocoYaml((*map)["yaml"].data())) {
        free(*ptr);
        *ptr = nullptr;
        return -1;
    }
#endif
    (*ptr)->yoloDetector.classes = s_classes;

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = (*map)["model"].data();
    params.imgSize = {640, 640};
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif
    auto str = (*ptr)->yoloDetector.CreateSession(params);
    if (str && e) {
        *e = str;
    }
    if (!str) {
        LOG_I("create plugin instance success: %s", s_plugin_name());
    }
    return str ? -1 : 0;
}

static void s_plugin_instance_free(plugin_instance **ptr) {
    if (!ptr || !*ptr) {
        return;
    }
    free(*ptr);
    *ptr = nullptr;
    LOG_I("free plugin instance: %s", s_plugin_name());
}

static AVPixelFormat s_plugin_input_pixel_fmt(plugin_instance *ptr) {
    return AV_PIX_FMT_RGB24;
}

static int s_plugin_instance_input(plugin_instance *ptr, AVFrame *frame, void *out) {
    assert(ptr && frame && frame->format == s_plugin_input_pixel_fmt(ptr));
    return ptr->Detector(frame);
}

static plugin_interface interface{
        .plugin_name = s_plugin_name,
        .plugin_onload = s_plugin_onload,
        .plugin_onunload = s_plugin_onunload,
        .plugin_max_threads = s_plugin_max_threads,
        .plugin_instance_create = s_plugin_instance_create,
        .plugin_instance_free = s_plugin_instance_free,
        .plugin_input_pixel_fmt = s_plugin_input_pixel_fmt,
        .plugin_instance_input = s_plugin_instance_input,
};

API_EXPORT const plugin_interface *get_plugin_interface() {
    return &interface;
}