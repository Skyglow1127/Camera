#include "vision_io.hpp"
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace grasp {
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
std::string parent(const std::string& p) { auto n=p.find_last_of('/'); return n==std::string::npos?".":n==0?"/":p.substr(0,n); }
std::string resolve(const std::string& dir,const std::string& p) { return p.empty() || p[0]=='/'?p:dir+"/"+p; }
void makeDirectory(const std::string& path) {
    if(path.empty()) throw std::runtime_error("Empty output directory");
    if(path!="/" && path!=".") makeDirectory(parent(path));
    if(::mkdir(path.c_str(),0755)!=0 && errno!=EEXIST) throw std::runtime_error("Cannot create "+path);
    struct stat info{};
    if(::stat(path.c_str(),&info)!=0 || !S_ISDIR(info.st_mode)) throw std::runtime_error("Not a directory: "+path);
}
template<class T> static void read(const cv::FileStorage& f,const char* key,T& value) { if(!f[key].empty()) f[key]>>value; }
Config loadConfig(const std::string& path) {
    cv::FileStorage f(path,cv::FileStorage::READ);
    if(!f.isOpened()) throw std::runtime_error("Cannot open config: "+path);
    Config c; c.weights="yolov4.weights"; c.model="yolov4.cfg"; c.names="coco.names"; c.output="output";
    read(f,"weights",c.weights); read(f,"model",c.model); read(f,"names",c.names); read(f,"output",c.output);
    read(f,"detection_mode",c.mode);
    if(c.mode!="tabletop" && c.mode!="yolo" && c.mode!="segment") throw std::runtime_error("detection_mode must be tabletop, yolo or segment");
    read(f,"segmentation_model",c.onnx); read(f,"segmentation_names",c.segmentNames);
    read(f,"handeye_file",c.handeye); read(f,"robot_pose_file",c.pose);
    read(f,"input_size",c.input); read(f,"cpu_threads",c.threads); read(f,"color_width",c.width);
    read(f,"color_height",c.height); read(f,"fps",c.fps); read(f,"detection_interval_ms",c.interval);
    read(f,"continuous",c.continuous); read(f,"letterbox",c.letterbox);
    read(f,"confidence",c.confidence); read(f,"nms_iou",c.nms); read(f,"max_rgbd_skew_ms",c.syncMs);
    read(f,"max_export_age_ms",c.maxExportAgeMs); read(f,"pose_tolerance_ms",c.poseToleranceMs);
    if(!f["classes"].empty()) f["classes"]>>c.classes;
    if(!f["excluded_grasp_classes"].empty()) f["excluded_grasp_classes"]>>c.excludedGraspClasses;
    read(f,"min_depth_m",c.geometry.minDepth); read(f,"max_depth_m",c.geometry.maxDepth);
    read(f,"plane_threshold_m",c.geometry.planeThreshold); read(f,"min_object_height_m",c.geometry.minHeight);
    read(f,"max_object_height_m",c.geometry.maxHeight); read(f,"min_plane_ratio",c.geometry.minPlaneRatio);
    read(f,"min_valid_depth_ratio",c.geometry.minValidRatio); read(f,"max_gripper_opening_m",c.geometry.maxOpening);
    read(f,"finger_margin_m",c.geometry.fingerMargin); read(f,"min_object_pixels",c.geometry.minObjectPixels);
    if(c.input<128 || c.input>1280 || c.input%32 || c.threads<1 || c.width<1 || c.height<1 || c.fps<1 || c.interval<0 ||
       !(c.confidence>0 && c.confidence<1) || !(c.nms>0 && c.nms<1) || !(c.syncMs>0) || !(c.maxExportAgeMs>0) ||
       !(c.poseToleranceMs>0) || !(c.geometry.minDepth>0 && c.geometry.maxDepth>c.geometry.minDepth) ||
       !(c.geometry.planeThreshold>0) || !(c.geometry.minHeight>0 && c.geometry.maxHeight>c.geometry.minHeight) ||
       !(c.geometry.minPlaneRatio>0 && c.geometry.minPlaneRatio<=1) || !(c.geometry.minValidRatio>0 && c.geometry.minValidRatio<=1) ||
       !(c.geometry.maxOpening>0) || !(c.geometry.fingerMargin>=0) || c.geometry.minObjectPixels<5)
        throw std::runtime_error("Invalid configuration values");
    std::string dir=parent(path);
    c.weights=resolve(dir,c.weights); c.model=resolve(dir,c.model); c.names=resolve(dir,c.names);
    c.output=resolve(dir,c.output); c.handeye=resolve(dir,c.handeye); c.pose=resolve(dir,c.pose);
    c.onnx=resolve(dir,c.onnx); c.segmentNames=resolve(dir,c.segmentNames);
    return c;
}
void attachBasePose(Result& r,const Config& c) {
    r.baseValid=false;
    if(c.handeye.empty() || c.pose.empty()) return;
    r.baseReason="invalid_calibration_or_pose";
    try {
        cv::FileStorage h(c.handeye,cv::FileStorage::READ),p(c.pose,cv::FileStorage::READ);
        if(!h.isOpened() || !p.isOpened()) return;
        int verified=0,stationary=0; std::string domain,units,poseUnits;
        h["verified"]>>verified; h["translation_unit"]>>units;
        p["stationary"]>>stationary; p["clock_domain"]>>domain; p["translation_unit"]>>poseUnits;
        if(!verified || units!="m" || poseUnits!="m") { r.baseReason="unverified_calibration_or_wrong_units"; return; }
        if(!stationary) { r.baseReason="stationary_robot_required"; return; }
        // SDK host timestamp is RECEIPT time, not exposure time. Stationary capture only.
        double time=0; p["host_received_ms"]>>time;
        if(domain!="sdk_host_received_ms" || !std::isfinite(time) || time<=0 ||
           std::abs(time-r.frame->hostReceivedMs)>c.poseToleranceMs) { r.baseReason="robot_pose_timestamp_mismatch"; return; }
        if(!r.frame->synchronized) { r.baseReason="rgbd_not_synchronized"; return; }
        cv::Mat he,pose; h["T_end_camera"]>>he; p["T_base_end"]>>pose;
        r.baseCamera=asTransform(pose)*asTransform(he);
        r.baseValid=true; r.baseReason="stationary_pose_only_not_motion_command";
    } catch(const std::exception& e) { r.baseReason=std::string("pose_error: ")+e.what(); }
}
void writeResult(const std::string& file,const Result& r,const std::vector<std::string>& names,bool offline,int selected) {
    cv::FileStorage f(file+".tmp",cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    if(!f.isOpened()) throw std::runtime_error("Cannot write "+file);
    f<<"schema_version"<<1<<"robot"<<"Unitree D1-T"<<"motion_command_ready"<<0<<"offline_replay"<<int(offline);
    f<<"detection_mode"<<r.mode<<"confidence_semantics"<<(r.mode=="tabletop"?"geometry_component_not_class_probability":"model_class_score");
    f<<"frame_id"<<std::to_string(r.frame->id)<<"color_device_us"<<r.frame->deviceColorUs<<"depth_device_us"<<r.frame->deviceDepthUs;
    f<<"host_received_ms"<<r.frame->hostReceivedMs<<"depth_host_received_ms"<<r.frame->depthHostReceivedMs
     <<"rgbd_pair_accepted"<<int(r.frame->synchronized)<<"rgbd_pairing"<<r.frame->pairing;
    f<<"frame_age_at_write_ms"<<elapsed(r.frame->received)<<"inference_ms"<<r.inferenceMs<<"preprocess_ms"<<r.preprocessMs;
    f<<"fresh_at_write"<<int(elapsed(r.frame->received)<=r.maxAgeMs)<<"max_result_age_ms"<<r.maxAgeMs;
    f<<"geometry_ms"<<r.geometryMs<<"processing_ms"<<r.totalMs<<"decode_ms"<<r.frame->decodeMs;
    f<<"translation_unit"<<"m"<<"camera_axes"<<"x_right_y_down_z_forward";
    f<<"pose_semantics"<<"surface_candidate: x_closing_axis, z_approach; TCP/contact offset and collisions NOT solved";
    f<<"base_transform_valid"<<int(r.baseValid)<<"base_transform_status"<<r.baseReason<<"selected_index"<<selected;
    f<<"table"<<"{"<<"valid"<<int(r.plane.valid)<<"normal_camera"<<cv::Mat(r.plane.normal)
     <<"offset_m"<<r.plane.offset<<"inlier_ratio"<<r.plane.inlierRatio<<"rms_m"<<r.plane.rms<<"}";
    f<<"candidates"<<"[";
    for(auto& c:r.candidates) {
        f<<"{"<<"class"<<names.at(c.detection.classId)<<"confidence"<<c.detection.score
         <<"box_xywh"<<"["<<c.detection.box.x<<c.detection.box.y<<c.detection.box.width<<c.detection.box.height<<"]"
         <<"valid"<<int(c.valid)<<"status"<<c.reason<<"depth_valid_ratio"<<c.validRatio;
        f<<"depth_measurement_valid"<<int(c.measured);
        if(c.measured) f<<"surface_pixel"<<"["<<c.pixel.x<<c.pixel.y<<"]"<<"surface_camera_m"<<cv::Mat(c.surface);
        if(c.valid) f<<"estimated_opening_m"<<c.opening<<"height_above_table_m"<<c.height<<"local_depth_mad_m"<<c.depthMad
          <<"orientation_anisotropy"<<c.orientationRatio<<"T_camera_surface"<<cv::Mat(surfaceTransform(c));
        if(c.valid && r.baseValid) f<<"T_base_surface"<<cv::Mat(r.baseCamera*surfaceTransform(c));
        f<<"}";
    }
    f<<"]"; f.release();
    if(std::rename((file+".tmp").c_str(),file.c_str())!=0) throw std::runtime_error("Cannot replace "+file);
}
void savePacket(const std::string& stem,const Packet& p,const Camera& camera) {
    if(!cv::imwrite(stem+"_color.png",p.color)) throw std::runtime_error("Cannot save color snapshot");
    cv::Mat mm; p.depth.convertTo(mm,CV_16U,1000);
    if(!cv::imwrite(stem+"_depth.png",mm)) throw std::runtime_error("Cannot save depth snapshot");
    cv::FileStorage f(stem+".yml",cv::FileStorage::WRITE);
    if(!f.isOpened()) throw std::runtime_error("Cannot save metadata");
    auto name=[](const std::string& path){return path.substr(path.find_last_of('/')+1);};
    f<<"color"<<name(stem+"_color.png")<<"depth"<<name(stem+"_depth.png")<<"depth_unit_m"<<0.001
     <<"aligned_to_color"<<1<<"width"<<camera.size.width<<"height"<<camera.size.height
     <<"camera_matrix"<<cv::Mat(camera.K)<<"distortion"<<camera.distortion
     <<"host_received_ms"<<p.hostReceivedMs<<"depth_host_received_ms"<<p.depthHostReceivedMs
     <<"rgbd_pairing"<<p.pairing<<"color_device_us"<<p.deviceColorUs<<"depth_device_us"<<p.deviceDepthUs;
}
std::shared_ptr<Packet> loadPacket(const std::string& path,Camera& camera,const Config& config) {
    cv::FileStorage f(path,cv::FileStorage::READ);
    if(!f.isOpened()) throw std::runtime_error("Cannot open replay metadata");
    int aligned=0; double unit=0; std::string color,depth;
    f["aligned_to_color"]>>aligned; f["depth_unit_m"]>>unit; f["color"]>>color; f["depth"]>>depth;
    if(!aligned || !std::isfinite(unit) || unit<=0) throw std::runtime_error("Replay needs aligned depth and positive depth_unit_m");
    auto p=std::make_shared<Packet>(); p->id=1;
    p->color=cv::imread(resolve(parent(path),color),cv::IMREAD_COLOR);
    cv::Mat raw=cv::imread(resolve(parent(path),depth),cv::IMREAD_UNCHANGED);
    if(p->color.empty() || raw.type()!=CV_16UC1 || raw.size()!=p->color.size()) throw std::runtime_error("Invalid replay images");
    raw.convertTo(p->depth,CV_32F,unit);
    f["width"]>>camera.size.width; f["height"]>>camera.size.height;
    cv::Mat k; f["camera_matrix"]>>k;
    if(k.rows!=3 || k.cols!=3 || k.channels()!=1) throw std::runtime_error("camera_matrix must be 3x3");
    k.convertTo(k,CV_64F); std::copy(k.ptr<double>(),k.ptr<double>()+9,camera.K.val);
    f["distortion"]>>camera.distortion; camera.validate();
    if(camera.size!=p->color.size()) throw std::runtime_error("Replay intrinsics resolution mismatch");
    f["host_received_ms"]>>p->hostReceivedMs; f["color_device_us"]>>p->deviceColorUs; f["depth_device_us"]>>p->deviceDepthUs;
    read(f,"depth_host_received_ms",p->depthHostReceivedMs); read(f,"rgbd_pairing",p->pairing);
    p->synchronized=p->deviceColorUs>0 && p->deviceDepthUs>0 && std::isfinite(p->deviceColorUs) &&
        std::isfinite(p->deviceDepthUs) && std::abs(p->deviceColorUs-p->deviceDepthUs)<=config.syncMs*1000;
    if(p->pairing=="host_receipt_approximate_stationary_only")
        p->synchronized=p->hostReceivedMs>0 && p->depthHostReceivedMs>0 &&
            std::isfinite(p->hostReceivedMs) && std::isfinite(p->depthHostReceivedMs) &&
            std::abs(p->hostReceivedMs-p->depthHostReceivedMs)<=config.syncMs;
    else if(p->pairing!="device_timestamp") throw std::runtime_error("Unsupported replay timestamp clock");
    return p;
}
cv::Mat drawResult(const Result& r,const std::vector<std::string>& names,int selected,const Camera* camera) {
    cv::Mat image=r.frame->color.clone();
    for(size_t i=0;i<r.candidates.size();++i) {
        auto& c=r.candidates[i]; cv::Scalar color=c.valid?cv::Scalar(0,220,0):cv::Scalar(0,160,255);
        if(int(i)==selected) color=cv::Scalar(255,220,0);
        cv::rectangle(image,c.detection.box,color,int(i)==selected?3:1);
        if(!c.detection.mask.empty()) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(c.detection.mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(image,contours,-1,color,1);
        }
        std::string text=std::to_string(i)+" "+names[c.detection.classId]+" "+cv::format("%.2f",c.detection.score);
        if(c.measured) {
            cv::drawMarker(image,c.pixel,color,cv::MARKER_CROSS,14,2);
            text+=cv::format(" Z=%.3fm",c.surface[2]);
            if(c.valid) text+=cv::format(" W=%.0fmm",1000*c.opening);
            if(camera && c.valid) {
                cv::Vec3d axis(c.axes(0,0),c.axes(1,0),c.axes(2,0));
                cv::Vec3d a=c.surface-axis*c.opening/2,b=c.surface+axis*c.opening/2;
                std::vector<cv::Point3d> points{{a[0],a[1],a[2]},{b[0],b[1],b[2]}};
                std::vector<cv::Point2d> projected;
                cv::projectPoints(points,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),camera->K,camera->distortion,projected);
                cv::arrowedLine(image,projected[0],projected[1],color,2,cv::LINE_AA,0,0.2);
            }
        }
        cv::putText(image,text,{std::max(0,c.detection.box.x),std::max(45,c.detection.box.y-6)},cv::FONT_HERSHEY_SIMPLEX,0.43,color,1);
        if(!c.valid) cv::putText(image,c.reason,{c.detection.box.x,std::min(image.rows-8,c.detection.box.y+17)},cv::FONT_HERSHEY_SIMPLEX,0.4,color,1);
    }
    cv::rectangle(image,{0,0,image.cols,36},cv::Scalar(0,0,0),cv::FILLED);
    cv::putText(image,cv::format("%s | age %.0fms | process %.0fms | table %s",r.mode.c_str(),elapsed(r.frame->received),r.totalMs,r.plane.valid?"OK":"INVALID"),
                {8,23},cv::FONT_HERSHEY_SIMPLEX,0.45,cv::Scalar(255,255,255),1);
    return image;
}
cv::Mat depthOverlay(const Packet& p,const Config& c) {
    cv::Mat intensity,color,blended,view=p.color.clone();
    p.depth.convertTo(intensity,CV_8U,255.0/(c.geometry.maxDepth-c.geometry.minDepth),-255.0*c.geometry.minDepth/(c.geometry.maxDepth-c.geometry.minDepth));
    cv::applyColorMap(intensity,color,cv::COLORMAP_TURBO);
    cv::addWeighted(p.color,0.5,color,0.5,0,blended);
    cv::Mat valid=(p.depth>=c.geometry.minDepth)&(p.depth<=c.geometry.maxDepth);
    blended.copyTo(view,valid);
    cv::putText(view,"ALIGNED RGB-D | blue=near red=far | missing depth keeps RGB",{8,22},cv::FONT_HERSHEY_SIMPLEX,0.4,{255,255,255},1);
    return view;
}
}
