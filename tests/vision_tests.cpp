#include "vision_core.hpp"
#include "vision_io.hpp"
#include "handeye.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace grasp;
static int checks=0;
static void check(bool condition,const std::string& message) {
    ++checks; if(!condition) throw std::runtime_error(message);
}
static cv::Matx44d pose(cv::Vec3d axis,cv::Vec3d translation) {
    cv::Mat R; cv::Rodrigues(axis,R); cv::Matx44d t=cv::Matx44d::eye();
    for(int i=0;i<3;++i) { t(i,3)=translation[i]; for(int j=0;j<3;++j) t(i,j)=R.at<double>(i,j); }
    return t;
}
int main() {
    try {
        Camera camera; camera.size={160,120}; camera.K={200,0,80,0,200,60,0,0,1};
        cv::Mat rays=makeRays(camera),depth(camera.size,CV_32F,cv::Scalar(0.8));
        Detection det{39,0.95f,{55,40,50,40}};
        depth(cv::Rect(60,45,40,30)).setTo(0.7);
        depth.at<float>(60,80)=0; // Missing center must not become an invented surface point.
        depth.at<float>(50,65)=std::numeric_limits<float>::quiet_NaN();
        Settings settings; settings.maxOpening=0.15;
        auto plane=fitTable(depth,rays,{det},settings);
        check(plane.valid,"table plane should be found");
        check(std::abs(plane.offset-0.8)<1e-5,"table depth must be in metres");
        auto objects=tableObjects(depth,rays,plane,settings);
        check(objects.size()==1,"geometry mode finds a separated unknown object without YOLO");
        auto candidate=estimate(det,depth,rays,plane,settings);
        check(candidate.valid,"isolated object must produce a candidate: "+candidate.reason);
        Plane absent;
        auto measurement=estimate(det,depth,rays,absent,settings);
        check(measurement.measured && !measurement.valid,"depth observation must be independent of grasp validity");
        check(std::abs(candidate.surface[2]-0.7)<1e-5,"surface should exclude tabletop");
        check(std::abs(candidate.height-0.1)<1e-5,"object height above table");
        Detection masked=det; masked.mask=cv::Mat(camera.size,CV_8U,cv::Scalar(0));
        masked.mask(cv::Rect(60,45,40,30)).setTo(255);
        auto maskedCandidate=estimate(masked,depth,rays,plane,settings);
        check(maskedCandidate.valid && maskedCandidate.validRatio>0.9,"instance-mask depth support excludes background pixels");
        check(candidate.opening>0.10 && candidate.opening<0.12,"metric minor-axis opening");
        check(depth.at<float>(candidate.pixel)>0,"surface point must have measured depth");
        check(validTransform(cv::Mat(surfaceTransform(candidate))),"candidate axes must form a proper rotation");
        check(cv::norm(pointAt(rays,80,60,0.7)-cv::Vec3d(0,0,0.7))<1e-7,"deprojection center");
        settings.maxOpening=0.06;
        check(!estimate(det,depth,rays,plane,settings).valid,"reject excessive opening");
        cv::Mat missing(camera.size,CV_32F,cv::Scalar(0));
        check(!fitTable(missing,rays,{det},settings).valid,"empty depth cannot produce a plane");
        check(!estimate(det,missing,rays,plane,settings).valid,"empty object depth cannot produce grasp");
        cv::Mat flat(camera.size,CV_32F,cv::Scalar(0.8));
        check(!estimate(det,flat,rays,plane,settings).valid,"tabletop itself is not an object");
        bool rejected=false;
        try { estimate(det,cv::Mat(10,10,CV_32F),rays,plane,settings); } catch(const std::exception&) { rejected=true; }
        check(rejected,"reject unaligned depth dimensions");
        auto kept=classNms({{0,0.9f,{10,10,40,40}},{0,0.8f,{11,11,40,40}},{1,0.7f,{10,10,40,40}}},0.3,0.45);
        check(kept.size()==2,"NMS must keep overlapping different classes");
        cv::Mat bad=cv::Mat::eye(4,4,CV_64F); bad.at<double>(0,0)=2;
        check(!validTransform(bad),"reject nonrigid transform");
        bad=cv::Mat::eye(4,4,CV_64F); bad.at<double>(0,0)=-1;
        check(!validTransform(bad),"reject reflection");
        bad=cv::Mat::eye(4,4,CV_64F); bad.at<double>(0,3)=std::numeric_limits<double>::quiet_NaN();
        check(!validTransform(bad),"reject NaN transform");
        // Synthetic eye-in-hand observations from a fixed target, independently generated poses.
        auto truth=pose({0.2,-0.1,0.15},{0.04,-0.02,0.08});
        auto fixedTarget=pose({0.1,0.2,-0.3},{0.5,0.1,0.2});
        std::vector<cv::Matx44d> robot,target;
        for(int i=0;i<18;++i) {
            auto b=pose({0.6*std::sin(i*0.7),0.5*std::cos(i*0.5),0.4*std::sin(i*0.3)},
                        {0.2+0.04*std::sin(i),0.08*std::cos(i),0.6+0.03*i});
            robot.push_back(b); target.push_back(truth.inv()*b.inv()*fixedTarget);
        }
        auto calibration=solveHandEye(robot,target);
        check(cv::norm(cv::Mat(calibration.endCamera-truth))<1e-6,"recover known eye-in-hand transform");
        check(calibration.translationRms<1e-7,"fixed-target consistency");
        rejected=false;
        try { solveHandEye(std::vector<cv::Matx44d>(8,cv::Matx44d::eye()),std::vector<cv::Matx44d>(8,cv::Matx44d::eye())); }
        catch(const std::exception&) { rejected=true; }
        check(rejected,"reject degenerate hand-eye motion");
        // Leave an offline fixture for the real model smoke test. Synthetic image has no promised detections.
        makeDirectory("test_artifacts");
        Packet p; p.color=cv::Mat(camera.size,CV_8UC3,cv::Scalar(110,110,110)); p.depth=depth;
        p.deviceColorUs=1000000; p.deviceDepthUs=1001000; p.hostReceivedMs=10000; p.depthHostReceivedMs=10001; p.synchronized=true;
        savePacket("test_artifacts/synthetic",p,camera);
        Camera loaded; Config config;
        auto replay=loadPacket("test_artifacts/synthetic.yml",loaded,config);
        check(replay->synchronized && replay->depth.size()==depth.size(),"snapshot roundtrip");
        check(std::abs(replay->depth.at<float>(46,61)-0.7)<1e-5,"millimetre replay scaling");
        cv::FileStorage h("test_artifacts/handeye.yml",cv::FileStorage::WRITE);
        h<<"translation_unit"<<"m"<<"verified"<<1<<"T_end_camera"<<cv::Mat(truth); h.release();
        auto robotPose=pose({0.1,0.3,-0.1},{0.3,0.2,0.1});
        auto writePose=[&](double time,int stationary) {
            cv::FileStorage f("test_artifacts/pose.yml",cv::FileStorage::WRITE);
            f<<"translation_unit"<<"m"<<"clock_domain"<<"sdk_host_received_ms"<<"stationary"<<stationary
             <<"host_received_ms"<<time<<"T_base_end"<<cv::Mat(robotPose);
        };
        config.handeye="test_artifacts/handeye.yml"; config.pose="test_artifacts/pose.yml";
        writePose(10000,1); Result r; r.frame=replay; attachBasePose(r,config);
        check(r.baseValid && cv::norm(cv::Mat(r.baseCamera-robotPose*truth))<1e-8,"correct base/end/camera transform order");
        writePose(9000,1); Result stale; stale.frame=replay; attachBasePose(stale,config);
        check(!stale.baseValid,"stale robot pose must be rejected");
        writePose(10000,0); Result moving; moving.frame=replay; attachBasePose(moving,config);
        check(!moving.baseValid,"moving robot must not use stationary pose interface");
        cv::FileStorage samples("test_artifacts/calibration_samples.yml",cv::FileStorage::WRITE);
        samples<<"translation_unit"<<"m"<<"samples"<<"[";
        for(size_t i=0;i<robot.size();++i) samples<<"{"<<"T_base_end"<<cv::Mat(robot[i])<<"T_camera_target"<<cv::Mat(target[i])<<"}";
        samples<<"]";
        std::cout<<checks<<" checks passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAILED after "<<checks<<" checks: "<<e.what()<<"\n"; return 1; }
}
