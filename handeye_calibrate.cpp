#include "handeye.hpp"
#include "vision_io.hpp"
#include <iostream>
#include <cmath>

int main(int argc,char** argv) {
    try {
        if(argc!=3 || std::string(argv[1])=="--help") {
            std::cout<<"handeye_calibrate samples.yml output.yml\nEach sample: T_base_end plus either T_camera_target or snapshot path. Units: metres. Fixed chessboard, stationary robot.\n";
            return argc==2?0:1;
        }
        cv::FileStorage input(argv[1],cv::FileStorage::READ);
        if(!input.isOpened()) throw std::runtime_error("Cannot open samples file");
        std::string unit; input["translation_unit"]>>unit;
        if(unit!="m") throw std::runtime_error("translation_unit must be m");
        int cols=0,rows=0; double square=0;
        input["board_columns"]>>cols; input["board_rows"]>>rows; input["square_size_m"]>>square;
        std::vector<cv::Matx44d> baseEnd,cameraTarget;
        double maxReprojection=0;
        for(auto sample:input["samples"]) {
            cv::Mat robot,target; sample["T_base_end"]>>robot;
            if(sample["T_camera_target"].empty()) {
                if(cols<3 || rows<3 || cols>100 || rows>100 || !std::isfinite(square) || square<=0) throw std::runtime_error("Invalid chessboard inner-corner dimensions/square size");
                std::string path; sample["snapshot"]>>path;
                grasp::Camera camera; grasp::Config config;
                auto packet=grasp::loadPacket(grasp::resolve(grasp::parent(argv[1]),path),camera,config);
                cv::Mat gray; cv::cvtColor(packet->color,gray,cv::COLOR_BGR2GRAY);
                std::vector<cv::Point2f> corners;
                if(!cv::findChessboardCornersSB(gray,{cols,rows},corners,cv::CALIB_CB_NORMALIZE_IMAGE|cv::CALIB_CB_EXHAUSTIVE))
                    throw std::runtime_error("Chessboard not found in "+path);
                std::vector<cv::Point3f> board;
                for(int y=0;y<rows;++y) for(int x=0;x<cols;++x) board.emplace_back(x*square,y*square,0);
                cv::Mat rvec,tvec,R;
                if(!cv::solvePnP(board,corners,camera.K,camera.distortion,rvec,tvec)) throw std::runtime_error("PnP failed: "+path);
                cv::Rodrigues(rvec,R); target=cv::Mat::eye(4,4,CV_64F);
                R.copyTo(target(cv::Rect(0,0,3,3))); tvec.copyTo(target(cv::Rect(3,0,1,3)));
                std::vector<cv::Point2f> projected; cv::projectPoints(board,rvec,tvec,camera.K,camera.distortion,projected);
                double error=cv::norm(corners,projected,cv::NORM_L2)/std::sqrt(corners.size());
                maxReprojection=std::max(maxReprojection,error);
                if(error>2) throw std::runtime_error("Chessboard reprojection RMS exceeds 2 pixels: "+path);
            } else sample["T_camera_target"]>>target;
            baseEnd.push_back(grasp::asTransform(robot)); cameraTarget.push_back(grasp::asTransform(target));
        }
        auto result=grasp::solveHandEye(baseEnd,cameraTarget);
        cv::FileStorage output(argv[2],cv::FileStorage::WRITE);
        if(!output.isOpened()) throw std::runtime_error("Cannot write calibration output");
        output<<"translation_unit"<<"m"<<"verified"<<0<<"T_end_camera"<<cv::Mat(result.endCamera)
              <<"sample_count"<<int(baseEnd.size())<<"target_consistency_translation_rms_m"<<result.translationRms
              <<"target_consistency_rotation_rms_deg"<<result.rotationRmsDegrees<<"max_reprojection_rms_px"<<maxReprojection;
        std::cout<<"Wrote "<<argv[2]<<". Consistency RMS: "<<result.translationRms*1000<<" mm, "<<result.rotationRmsDegrees
                 <<" deg. verified=0 until independent physical validation.\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"Calibration error: "<<e.what()<<"\n"; return 1; }
}
