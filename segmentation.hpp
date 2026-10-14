#pragma once
#include "vision_io.hpp"
#include <memory>
namespace grasp {
class Segmenter {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    explicit Segmenter(const Config& config);
    ~Segmenter();
    std::vector<Detection> run(const cv::Mat& color,Result& result);
    std::vector<std::string> names;
};
}
