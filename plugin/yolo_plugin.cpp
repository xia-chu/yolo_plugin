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


static std::vector<cv::Scalar> s_classes_color;

static bool init_color() {
    cv::RNG rng(cv::getTickCount());
    s_classes_color.resize(s_classes.size());
    for (auto i = 0; i < s_classes.size(); ++i) {
        s_classes_color[i] = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    }
    return true;
}

struct plugin_instance {
    YOLO_V8 yoloDetector;
    int interval = 1;

    int Detector(size_t index, AVFrame *frame) {
        if (frame->format != AV_PIX_FMT_RGB24) {
            LOG_E("Input AVFrame is not in RGB24 format");
            return -1;
        }

        if (index % interval) {
            return 0;
        }

        cv::Mat img(frame->height, frame->width, CV_8UC3, frame->data[0], frame->linesize[0]);
        std::vector<DL_RESULT> res;
        yoloDetector.RunSession(img, res);
        for (auto &re: res) {
            auto &color = s_classes_color[re.classId];
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
    static auto flag = init_color();
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

static int s_plugin_max_thread_safety() {
    return false;
}

static int s_plugin_instance_create(plugin_instance **ptr, const char *const* config_map, char *err) {
    assert(ptr && config_map);
    *ptr = (plugin_instance *) new plugin_instance;

    std::map<std::string, std::string> map;
    for (auto i = 0u; config_map[i] && config_map[i + 1];) {
        map[config_map[i]] = config_map[i + 1];
        i += 2;
    }
    map.emplace("interval", "1");
    map.emplace("rectConfidenceThreshold", "0.1");
    map.emplace("iouThreshold", "0.5");
    map.emplace("model", "yolov8n.onnx");
    map.emplace("imgWidth", "640");
    map.emplace("imgHeight", "640");

    (*ptr)->yoloDetector.classes = s_classes;
    (*ptr)->interval = std::atoi(map["interval"].data());

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = std::atof(map["rectConfidenceThreshold"].data());
    params.iouThreshold = std::atof(map["iouThreshold"].data());
    params.modelPath = map["model"].data();
    params.imgSize = {std::atoi(map["imgWidth"].data()), std::atoi(map["imgHeight"].data())};
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
    if (str && err) {
        strcpy(err, str);
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
    delete *ptr;
    *ptr = nullptr;
    LOG_I("free plugin instance: %s", s_plugin_name());
}

static AVPixelFormat s_plugin_input_pixel_fmt(plugin_instance *ptr) {
    return AV_PIX_FMT_RGB24;
}

static int s_plugin_instance_input(plugin_instance *ptr, size_t index, AVFrame *frame, void *out) {
    assert(ptr && frame && frame->format == s_plugin_input_pixel_fmt(ptr));
    return ptr->Detector(index, frame);
}

static plugin_interface interface{
        .plugin_name = s_plugin_name,
        .plugin_onload = s_plugin_onload,
        .plugin_onunload = s_plugin_onunload,
        .plugin_max_threads = s_plugin_max_threads,
        .plugin_max_thread_safety = s_plugin_max_thread_safety,
        .plugin_instance_create = s_plugin_instance_create,
        .plugin_instance_free = s_plugin_instance_free,
        .plugin_input_pixel_fmt = s_plugin_input_pixel_fmt,
        .plugin_instance_input = s_plugin_instance_input,
};

API_EXPORT const plugin_interface *get_plugin_interface() {
    return &interface;
}