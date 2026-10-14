#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace grasp {
struct Settings {
    float minDepth = 0.15f, maxDepth = 1.5f;
    float planeThreshold = 0.006f, minHeight = 0.008f, maxHeight = 0.25f;
    float minPlaneRatio = 0.3f, minValidRatio = 0.25f;
    float maxOpening = 0.08f, fingerMargin = 0.005f;
    int minObjectPixels = 40;
};
struct Camera {
    cv::Size size;
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Mat distortion;
    void validate() const;
};
struct Detection { int classId; float score; cv::Rect box; cv::Mat mask{}; };
struct Plane {
    bool valid = false;
    cv::Vec3d normal{0, 0, -1}; // Points from the table towards the camera.
    double offset = 0, inlierRatio = 0, rms = 0;
};
struct Candidate {
    Detection detection;
    bool valid = false;
    bool measured = false;
    std::string reason = "no_depth";
    cv::Mat mask;
    cv::Point pixel{-1, -1};
    cv::Vec3d surface{0, 0, 0}; // A measured surface point, NOT the tool command position.
    cv::Matx33d axes = cv::Matx33d::eye(); // x: closing axis; z: approach towards table.
    double opening = 0, height = 0, validRatio = 0, depthMad = 0;
    double orientationRatio = 0; // Near 1: symmetric footprint; yaw is ambiguous.
};
cv::Mat makeRays(const Camera& camera);
cv::Vec3d pointAt(const cv::Mat& rays, int x, int y, double z);
Plane fitTable(const cv::Mat& depth, const cv::Mat& rays,
               const std::vector<Detection>& detections, const Settings& settings);
std::vector<Detection> tableObjects(const cv::Mat& depth,const cv::Mat& rays,const Plane& plane,const Settings& settings);
Candidate estimate(const Detection& detection, const cv::Mat& depth,
                   const cv::Mat& rays, const Plane& plane, const Settings& settings);
std::vector<Detection> classNms(const std::vector<Detection>& input, float score, float iou);
bool validTransform(const cv::Mat& matrix);
cv::Matx44d asTransform(const cv::Mat& matrix);
cv::Matx44d surfaceTransform(const Candidate& candidate);
double percentile(std::vector<double> values, double q);
} // namespace grasp
