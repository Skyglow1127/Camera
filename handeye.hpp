#pragma once
#include "vision_core.hpp"
namespace grasp {
struct HandEyeResult {
    cv::Matx44d endCamera;
    double translationRms=0,rotationRmsDegrees=0;
};
HandEyeResult solveHandEye(const std::vector<cv::Matx44d>& baseEnd,const std::vector<cv::Matx44d>& cameraTarget);
}
