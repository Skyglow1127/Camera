#include "handeye.hpp"
#include <cmath>
#include <stdexcept>
namespace grasp {
HandEyeResult solveHandEye(const std::vector<cv::Matx44d>& baseEnd,const std::vector<cv::Matx44d>& cameraTarget) {
    if(baseEnd.size()!=cameraTarget.size() || baseEnd.size()<8) throw std::runtime_error("Need at least 8 paired, diverse stationary poses (recommend 15-25)");
    std::vector<cv::Mat> robotR,robotT,targetR,targetT;
    cv::Mat motion(int(baseEnd.size())-1,3,CV_64F);
    for(size_t i=0;i<baseEnd.size();++i) {
        if(!validTransform(cv::Mat(baseEnd[i])) || !validTransform(cv::Mat(cameraTarget[i]))) throw std::runtime_error("Invalid calibration transform");
        cv::Mat b(baseEnd[i]),c(cameraTarget[i]);
        robotR.push_back(b(cv::Rect(0,0,3,3)).clone()); robotT.push_back(b(cv::Rect(3,0,1,3)).clone());
        targetR.push_back(c(cv::Rect(0,0,3,3)).clone()); targetT.push_back(c(cv::Rect(3,0,1,3)).clone());
        if(i>0) {
            cv::Mat rvec; cv::Rodrigues(robotR[0].t()*robotR[i],rvec);
            rvec.reshape(1,1).copyTo(motion.row(int(i)-1));
        }
    }
    cv::SVD diversity(motion);
    if(diversity.w.at<double>(1)<0.1) throw std::runtime_error("Insufficient rotation diversity: rotate about multiple nonparallel axes");
    cv::Mat R,t;
    cv::calibrateHandEye(robotR,robotT,targetR,targetT,R,t,cv::CALIB_HAND_EYE_PARK);
    cv::Mat transform=cv::Mat::eye(4,4,CV_64F);
    R.copyTo(transform(cv::Rect(0,0,3,3))); t.copyTo(transform(cv::Rect(3,0,1,3)));
    HandEyeResult result; result.endCamera=asTransform(transform);
    std::vector<cv::Matx44d> fixedTargets;
    cv::Vec3d mean{0,0,0}; cv::Mat averageRotation=cv::Mat::zeros(3,3,CV_64F);
    for(size_t i=0;i<baseEnd.size();++i) {
        auto pose=baseEnd[i]*result.endCamera*cameraTarget[i]; fixedTargets.push_back(pose);
        mean+=cv::Vec3d(pose(0,3),pose(1,3),pose(2,3));
        averageRotation+=cv::Mat(pose)(cv::Rect(0,0,3,3));
    }
    mean/=double(baseEnd.size());
    cv::SVD svd(averageRotation); cv::Mat correction=cv::Mat::eye(3,3,CV_64F);
    correction.at<double>(2,2)=cv::determinant(svd.u*svd.vt)<0?-1:1;
    cv::Mat meanR=svd.u*correction*svd.vt;
    for(auto& pose:fixedTargets) {
        cv::Vec3d delta=cv::Vec3d(pose(0,3),pose(1,3),pose(2,3))-mean;
        result.translationRms+=delta.dot(delta);
        cv::Mat rot=meanR.t()*cv::Mat(pose)(cv::Rect(0,0,3,3));
        double angle=std::acos(std::max(-1.0,std::min(1.0,(cv::trace(rot)[0]-1)/2)))*180/CV_PI;
        result.rotationRmsDegrees+=angle*angle;
    }
    result.translationRms=std::sqrt(result.translationRms/baseEnd.size());
    result.rotationRmsDegrees=std::sqrt(result.rotationRmsDegrees/baseEnd.size());
    return result;
}
}
