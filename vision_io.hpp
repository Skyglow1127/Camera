#pragma once
#include "vision_core.hpp"
#include <chrono>
#include <memory>

namespace grasp {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start);
std::string resolve(const std::string& directory,const std::string& path);
std::string parent(const std::string& path);
void makeDirectory(const std::string& path);
struct Config {
    std::string weights,model,names,output,handeye,pose,mode="tabletop";
    std::string onnx,segmentNames;
    int input=416,threads=4,width=640,height=480,fps=30,interval=0,continuous=1,letterbox=0;
    float confidence=0.35f,nms=0.45f;
    double syncMs=25,maxExportAgeMs=2000,poseToleranceMs=100;
    std::vector<std::string> classes;
    std::vector<std::string> excludedGraspClasses{"person"};
    Settings geometry;
};
Config loadConfig(const std::string& path);
struct Packet {
    uint64_t id=0;
    double deviceColorUs=0,deviceDepthUs=0,hostReceivedMs=0,depthHostReceivedMs=0,decodeMs=0;
    std::string pairing="device_timestamp";
    Clock::time_point received=Clock::now();
    cv::Mat color,depth,preview; // Owned pair; preview may be newer than the matched RGB-D color.
    bool synchronized=false;
};
struct Result {
    std::shared_ptr<Packet> frame;
    std::vector<Candidate> candidates;
    Plane plane;
    double preprocessMs=0,inferenceMs=0,geometryMs=0,totalMs=0;
    double maxAgeMs=2000;
    bool baseValid=false;
    std::string baseReason="robot_pose_not_configured";
    std::string mode="yolo";
    cv::Matx44d baseCamera=cv::Matx44d::eye();
};
void attachBasePose(Result& result,const Config& config);
void writeResult(const std::string& path,const Result& result,const std::vector<std::string>& names,bool offline,int selected=-1);
void savePacket(const std::string& stem,const Packet& packet,const Camera& camera);
std::shared_ptr<Packet> loadPacket(const std::string& path,Camera& camera,const Config& config);
cv::Mat drawResult(const Result& result,const std::vector<std::string>& names,int selected,const Camera* camera=nullptr);
cv::Mat depthOverlay(const Packet& packet,const Config& config);
}
