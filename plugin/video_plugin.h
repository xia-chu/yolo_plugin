//
// Created by xzl on 2025/1/4.
//

#ifndef PLUGIN_INTERFACE
#define PLUGIN_INTERFACE

#ifdef __cplusplus
extern "C" {
#endif

#define API_EXPORT __attribute__((visibility("default")))

#include "libavutil/frame.h"

typedef struct plugin_instance plugin_instance;

typedef struct {
    /**
     * 调用主框架的日志打印函数
     * @param level 日志等级
     * @param file 当前源码文件名
     * @param line 当前代码行
     * @param func 当前代码所在函数
     * @param fmt 格式化控制符
     * @param ... 参数列表
     */
    void (*printf_log)(int level, const char *file, int line, const char *func, const char *fmt, ...);
} plugin_loader;

typedef struct {
    /**
     * 获取插件名
     */
    const char *(*plugin_name)();

    /**
     * 插件加载后回调
     * @param loader 加载器
     */
    void (*plugin_onload)(const plugin_loader *loader);

    /**
     * 插件卸载前回调
     */
    void (*plugin_onunload)();

    /**
     * 该模型释放支持多线程并发执行
     */
    int (*plugin_max_threads)();

    /**
     * 创建插件实例
     * @param ptr 返回实例对象
     * @param config_map std::map<std::string, std::string> *类型，存放插件配置
     * @param err std::string* 类型，存放错误提示，可用为null
     * @return 0: 成功，其他为错误原因
     */
    int (*plugin_instance_create)(plugin_instance **ptr, void *config_map, void *err);

    /**
     * 释放插件实例
     * @param ptr 插件实例指针的指针
     */
    void (*plugin_instance_free)(plugin_instance **ptr);

    /**
     * 获取插件输入的图像格式
     * @param ptr 插件实例指针
     */
    AVPixelFormat (*plugin_input_pixel_fmt)(plugin_instance *ptr);

    /**
     * 输入视频帧并处理
     * @param ptr 插件实例指针
     * @param frame 输入的帧
     * @param out 处理后需要保存的结构化数据，可以为null
     */
    int (*plugin_instance_input)(plugin_instance *ptr, AVFrame *frame, void *out);
} plugin_interface;

/**
 * 获取插件入口
 */
API_EXPORT const plugin_interface* get_plugin_interface();


#ifdef __cplusplus
}
#endif

#endif //PLUGIN_INTERFACE
