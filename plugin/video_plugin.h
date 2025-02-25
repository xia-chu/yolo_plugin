//
// Created by xzl on 2025/1/4.
//

#ifndef PLUGIN_INTERFACE
#define PLUGIN_INTERFACE

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#   if defined(PLUGIN_EXPORTS)
#       define API_EXPORT __declspec(dllexport)
#   else
#       define API_EXPORT __declspec(dllimport)
#   endif
#   define API_CALL __cdecl
#else
#   define API_EXPORT __attribute__((visibility("default")))
#   define API_CALL
#endif

typedef struct AVFrame AVFrame;
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
    void (API_CALL *printf_log)(int level, const char *file, int line, const char *func, const char *fmt, ...);
} plugin_loader;

typedef struct {
    /**
     * 获取插件名
     */
    const char *(API_CALL *plugin_name)();

    /**
     * 插件加载后回调
     * @param loader 加载器
     */
    void (API_CALL *plugin_onload)(const plugin_loader *loader);

    /**
     * 插件卸载前回调
     */
    void (API_CALL *plugin_onunload)();

    /**
     * 该模型释放支持多线程并发执行
     */
    int (API_CALL *plugin_max_threads)();

    /**
     * 多线程模式下单实例是否线程安全
     */
    int (API_CALL *plugin_thread_safety)();

    /**
     * 创建插件实例
     * @param ptr 返回实例对象
     * @param config_map {"key0", "val0", "key1", "val2", NULL}类型，存放插件配置
     * @param err 存放错误提示，可用为null, 建议保留1024个字节
     * @return 0: 成功，其他为错误原因
     */
    int (API_CALL *plugin_instance_create)(plugin_instance **ptr, const char *const* config_map, char *err);

    /**
     * 释放插件实例
     * @param ptr 插件实例指针的指针
     */
    void (*plugin_instance_free)(plugin_instance **ptr);

    /**
     * 获取插件输入的图像格式
     * @param ptr 插件实例指针
     * @return 返回AVPixelFormat值
     */
    int (API_CALL *plugin_input_pixel_fmt)(plugin_instance *ptr);

    /**
     * 输入视频帧并处理
     * @param ptr 插件实例指针
     * @param frame_index 输入的帧序号
     * @param frame 输入的帧
     * @param out 处理后需要保存的结构化数据，可以为null
     */
    int (API_CALL *plugin_instance_input)(plugin_instance *ptr, int frame_index, AVFrame *frame, void *out);
} plugin_interface;

/**
 * 获取插件入口
 */
API_EXPORT const plugin_interface* API_CALL get_plugin_interface();


#ifdef __cplusplus
}
#endif

#endif //PLUGIN_INTERFACE
