#include "vision_io.hpp"
#ifdef HAVE_ONNXRUNTIME
#include "segmentation.hpp"
#endif
#include <libobsensor/ObSensor.hpp>
#include <atomic>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#ifndef CAMERA_ASSET_DIR
#define CAMERA_ASSET_DIR "."
#endif
using namespace grasp;
static volatile std::sig_atomic_t interrupted=0;
static void onSignal(int) { interrupted=1; }
class Detector {
    cv::dnn::Net net;
    std::vector<cv::String> outputs;
    Config config;
#ifdef HAVE_ONNXRUNTIME
    std::unique_ptr<Segmenter> segmenter;
#endif
public:
    std::vector<std::string> names;
    explicit Detector(const Config& c):config(c) {
        if(c.mode=="tabletop") { names={"object"}; return; }
        if(c.mode=="segment") {
#ifdef HAVE_ONNXRUNTIME
            segmenter.reset(new Segmenter(c)); names=segmenter->names; return;
#else
            throw std::runtime_error("Segmentation requires ONNX Runtime. Reconfigure CMake with ONNXRUNTIME_ROOT.");
#endif
        }
        std::ifstream f(c.names);
        if(!f) throw std::runtime_error("Cannot open class names: "+c.names);
        std::string name;
        while(std::getline(f,name)) { if(!name.empty() && name.back()=='\r') name.pop_back(); if(!name.empty()) names.push_back(name); }
        if(names.empty()) throw std::runtime_error("Empty class names");
        for(auto& wanted:c.classes) if(std::find(names.begin(),names.end(),wanted)==names.end()) throw std::runtime_error("Unknown target class: "+wanted);
        net=cv::dnn::readNetFromDarknet(c.model,c.weights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        outputs=net.getUnconnectedOutLayersNames();
    }
    std::vector<Detection> run(const cv::Mat& color,Result& result) {
#ifdef HAVE_ONNXRUNTIME
        if(segmenter) return segmenter->run(color,result);
#endif
        auto start=Clock::now(); cv::Mat input=color;
        double sx=double(config.input)/color.cols,sy=double(config.input)/color.rows;
        int px=0,py=0;
        if(config.letterbox) {
            double scale=std::min(sx,sy);
            cv::Size size(std::lround(color.cols*scale),std::lround(color.rows*scale));
            sx=double(size.width)/color.cols; sy=double(size.height)/color.rows;
            px=(config.input-size.width)/2; py=(config.input-size.height)/2;
            cv::Mat resized; cv::resize(color,resized,size);
            cv::copyMakeBorder(resized,input,py,config.input-size.height-py,px,config.input-size.width-px,cv::BORDER_CONSTANT,{114,114,114});
        }
        cv::Mat blob=cv::dnn::blobFromImage(input,1.0/255,cv::Size(config.input,config.input),cv::Scalar(),true,false);
        net.setInput(blob); result.preprocessMs=elapsed(start); start=Clock::now();
        std::vector<cv::Mat> out; net.forward(out,outputs); result.inferenceMs=elapsed(start);
        std::vector<Detection> raw;
        for(auto& m:out) {
            if(m.cols!=int(names.size())+5) throw std::runtime_error("Model class count does not match names");
            for(int i=0;i<m.rows;++i) {
                const float* row=m.ptr<float>(i); cv::Point cls; double score;
                cv::minMaxLoc(m.row(i).colRange(5,m.cols),nullptr,&score,nullptr,&cls);
                // Darknet Region output in OpenCV already includes objectness in class scores.
                if(!std::isfinite(score) || score<config.confidence) continue;
                if(!config.classes.empty() && std::find(config.classes.begin(),config.classes.end(),names[cls.x])==config.classes.end()) continue;
                double cx=(row[0]*config.input-px)/sx,cy=(row[1]*config.input-py)/sy;
                double w=row[2]*config.input/sx,h=row[3]*config.input/sy;
                if(!std::isfinite(cx+cy+w+h) || w<=0 || h<=0) continue;
                int x1=int(std::max(0.0,std::min(double(color.cols),std::floor(cx-w/2))));
                int y1=int(std::max(0.0,std::min(double(color.rows),std::floor(cy-h/2))));
                int x2=int(std::max(0.0,std::min(double(color.cols),std::ceil(cx+w/2))));
                int y2=int(std::max(0.0,std::min(double(color.rows),std::ceil(cy+h/2))));
                if(x2>x1 && y2>y1) raw.push_back({cls.x,float(score),cv::Rect(x1,y1,x2-x1,y2-y1)});
            }
        }
        return classNms(raw,config.confidence,config.nms);
    }
};
static Result analyze(const std::shared_ptr<Packet>& frame,Detector& detector,const cv::Mat& rays,const Config& c) {
    Result r; r.frame=frame; r.mode=c.mode; r.maxAgeMs=c.maxExportAgeMs; auto start=Clock::now();
    std::vector<Detection> detections;
    if(c.mode!="tabletop") detections=detector.run(frame->color,r);
    auto geometryStart=Clock::now();
    if(frame->synchronized) r.plane=fitTable(frame->depth,rays,detections,c.geometry);
    if(c.mode=="tabletop" && frame->synchronized) detections=tableObjects(frame->depth,rays,r.plane,c.geometry);
    for(auto& d:detections) {
        if(frame->synchronized) r.candidates.push_back(estimate(d,frame->depth,rays,r.plane,c.geometry));
        else { Candidate candidate; candidate.detection=d; candidate.reason="rgbd_timestamp_mismatch"; r.candidates.push_back(candidate); }
        if(std::find(c.excludedGraspClasses.begin(),c.excludedGraspClasses.end(),detector.names[d.classId])!=c.excludedGraspClasses.end()) {
            r.candidates.back().valid=false; r.candidates.back().reason="class_excluded_from_grasp";
        }
    }
    attachBasePose(r,c); r.geometryMs=elapsed(geometryStart); r.totalMs=elapsed(start); return r;
}
static bool supportedColor(OBFormat f) { return f==OB_FORMAT_RGB || f==OB_FORMAT_BGR || f==OB_FORMAT_MJPG || f==OB_FORMAT_YUYV || f==OB_FORMAT_YUY2 || f==OB_FORMAT_UYVY; }
// Y10/Y11/Y12/Y14 transport profiles are unpacked to 16-bit depth by SDK by default.
static bool supportedDepth(OBFormat f) { return f==OB_FORMAT_Y16 || f==OB_FORMAT_Z16 || f==OB_FORMAT_Y10 || f==OB_FORMAT_Y11 || f==OB_FORMAT_Y12 || f==OB_FORMAT_Y14; }
static cv::Mat decodeColor(const std::shared_ptr<ob::ColorFrame>& frame) {
    cv::Mat color; int w=frame->width(),h=frame->height(); auto fmt=frame->format();
    if(fmt==OB_FORMAT_MJPG) return cv::imdecode(cv::Mat(1,frame->dataSize(),CV_8U,frame->data()),cv::IMREAD_COLOR);
    size_t needed=size_t(w)*h*((fmt==OB_FORMAT_RGB || fmt==OB_FORMAT_BGR)?3:2);
    if(frame->dataSize()<needed) throw std::runtime_error("Truncated color frame");
    if(fmt==OB_FORMAT_BGR) return cv::Mat(h,w,CV_8UC3,frame->data()).clone();
    if(fmt==OB_FORMAT_RGB) cv::cvtColor(cv::Mat(h,w,CV_8UC3,frame->data()),color,cv::COLOR_RGB2BGR);
    else cv::cvtColor(cv::Mat(h,w,CV_8UC2,frame->data()),color,fmt==OB_FORMAT_UYVY?cv::COLOR_YUV2BGR_UYVY:cv::COLOR_YUV2BGR_YUY2);
    return color;
}
struct Shared {
    std::mutex mutex; std::condition_variable cv;
    std::shared_ptr<Packet> latest; std::shared_ptr<Result> result;
    std::string error;
    std::atomic<bool> stop{false},continuous{true},requested{false};
    std::atomic<uint64_t> captured{0},processed{0},unsynchronized{0};
};
static void fail(Shared& s,const std::string& error) {
    { std::lock_guard<std::mutex> lock(s.mutex); s.error=error; }
    s.stop=true; s.cv.notify_all();
}
struct Workers {
    Shared& s; std::thread capture,inference;
    explicit Workers(Shared& shared):s(shared) {}
    ~Workers() { s.stop=true; s.cv.notify_all(); if(capture.joinable()) capture.join(); if(inference.joinable()) inference.join(); }
};
int main(int argc,char** argv) {
    try {
        std::string configPath=std::string(CAMERA_ASSET_DIR)+"/grasp_config.yml",replay,modeOverride,modelOverride,outputOverride;
        bool headless=false,probe=false; double seconds=0; int benchmark=0;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            if(a=="--help") { std::cout<<"detect [--config FILE] [--mode segment|tabletop|yolo] [--segment-model FILE.onnx] [--output DIR]\n       [--replay SNAPSHOT.yml] [--benchmark N] [--headless] [--seconds N] [--probe]\nKeys: q/ESC quit, SPACE pause/resume inference, d detect once, TAB select, s save result, c capture RGB-D, v depth overlay.\n"; return 0; }
            else if(a=="--config" && i+1<argc) configPath=argv[++i];
            else if(a=="--replay" && i+1<argc) replay=argv[++i];
            else if(a=="--mode" && i+1<argc) modeOverride=argv[++i];
            else if(a=="--segment-model" && i+1<argc) modelOverride=argv[++i];
            else if(a=="--output" && i+1<argc) outputOverride=argv[++i];
            else if(a=="--seconds" && i+1<argc) { seconds=std::stod(argv[++i]); if(!std::isfinite(seconds) || seconds<=0) throw std::runtime_error("seconds must be positive"); }
            else if(a=="--headless") headless=true;
            else if(a=="--benchmark" && i+1<argc) { benchmark=std::stoi(argv[++i]); if(benchmark<1 || benchmark>2000) throw std::runtime_error("benchmark must be 1..2000"); }
            else if(a=="--probe") probe=true;
            else throw std::runtime_error("Unknown/incomplete argument: "+a);
        }
        Config c=loadConfig(configPath);
        if(!modeOverride.empty()) c.mode=modeOverride;
        if(c.mode!="segment" && c.mode!="tabletop" && c.mode!="yolo") throw std::runtime_error("Unknown detection mode");
        if(!modelOverride.empty()) { c.onnx=modelOverride; c.segmentNames=modelOverride.substr(0,modelOverride.find_last_of('.'))+".names"; }
        if(!outputOverride.empty()) c.output=outputOverride;
        cv::setNumThreads(c.threads); makeDirectory(c.output);
        if(benchmark && replay.empty()) throw std::runtime_error("--benchmark requires --replay");
        std::signal(SIGINT,onSignal); std::signal(SIGTERM,onSignal); Camera camera;
        if(!replay.empty()) {
            auto frame=loadPacket(replay,camera,c); Detector detector(c);
            cv::Mat rays=makeRays(camera);
            Result result;
            if(benchmark) {
                for(int i=0;i<3;++i) analyze(frame,detector,rays,c);
                std::vector<double> total,inference,geometry;
                for(int i=0;i<benchmark;++i) {
                    result=analyze(frame,detector,rays,c);
                    total.push_back(result.totalMs); inference.push_back(result.inferenceMs); geometry.push_back(result.geometryMs);
                }
                cv::FileStorage report(c.output+"/benchmark.json",cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
                report<<"mode"<<c.mode<<"model"<<(c.mode=="segment"?c.onnx:c.mode=="yolo"?c.weights:"none")
                      <<"input_size"<<c.input<<"cpu_threads"<<c.threads<<"warmup"<<3<<"iterations"<<benchmark
                      <<"total_median_ms"<<percentile(total,0.5)<<"total_p95_ms"<<percentile(total,0.95)
                      <<"inference_median_ms"<<percentile(inference,0.5)<<"geometry_median_ms"<<percentile(geometry,0.5)
                      <<"scope"<<"one recorded frame; processing only, excludes capture/display; not an accuracy evaluation";
                std::cout<<"Benchmark total median="<<percentile(total,0.5)<<"ms p95="<<percentile(total,0.95)<<"ms\n";
            } else result=analyze(frame,detector,rays,c);
            writeResult(c.output+"/replay_result.json",result,detector.names,true);
            cv::imwrite(c.output+"/replay_annotated.png",drawResult(result,detector.names,-1,&camera));
            for(size_t i=0;i<result.candidates.size();++i) if(!result.candidates[i].detection.mask.empty())
                cv::imwrite(c.output+"/replay_mask_"+std::to_string(i)+".png",result.candidates[i].detection.mask);
            std::cout<<"Replay complete: "<<c.output<<"/replay_result.json | infer "<<result.inferenceMs<<" ms\n"; return 0;
        }
        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
        ob::Context context; auto devices=context.queryDeviceList();
        if(!devices->deviceCount()) throw std::runtime_error("No Orbbec device found. Use --replay for offline testing.");
        auto device=devices->getDevice(0); ob::Pipeline pipe(device);
        std::cout<<"Camera: "<<device->getDeviceInfo()->name()<<"\n";
        auto colors=pipe.getStreamProfileList(OB_SENSOR_COLOR);
        if(probe) {
            auto depths=pipe.getStreamProfileList(OB_SENSOR_DEPTH);
            for(uint32_t i=0;i<depths->count();++i) {
                auto p=depths->getProfile(i)->as<ob::VideoStreamProfile>();
                std::cout<<"depth "<<p->width()<<"x"<<p->height()<<" @"<<p->fps()<<" format="<<p->format()<<"\n";
            }
        }
        std::vector<std::shared_ptr<ob::VideoStreamProfile>> available;
        for(uint32_t i=0;i<colors->count();++i) {
            auto p=colors->getProfile(i)->as<ob::VideoStreamProfile>();
            if(probe) std::cout<<"color "<<p->width()<<"x"<<p->height()<<" @"<<p->fps()<<" format="<<p->format()<<"\n";
            if(supportedColor(p->format())) available.push_back(p);
        }
        auto score=[&](const std::shared_ptr<ob::VideoStreamProfile>& p) {
            return std::abs(int(p->width())-c.width)+std::abs(int(p->height())-c.height)+20*std::abs(int(p->fps())-c.fps);
        };
        std::stable_sort(available.begin(),available.end(),[&](const auto&a,const auto&b){return score(a)<score(b);});
        std::shared_ptr<ob::VideoStreamProfile> colorProfile,depthProfile;
        OBAlignMode alignMode=ALIGN_DISABLE;
        for(auto& color:available) {
          for(auto mode:{ALIGN_D2C_HW_MODE,ALIGN_D2C_SW_MODE}) {
            try {
                auto depths=pipe.getD2CDepthProfileList(color,mode); int best=INT_MAX;
                for(uint32_t i=0;i<depths->count();++i) {
                    auto d=depths->getProfile(i)->as<ob::VideoStreamProfile>();
                    if(!supportedDepth(d->format()) || d->fps()!=color->fps()) continue;
                    int value=std::abs(int(d->width())-c.width)+std::abs(int(d->height())-c.height);
                    if(value<best) { depthProfile=d; best=value; }
                }
                if(best!=INT_MAX) { colorProfile=color; alignMode=mode; break; }
            } catch(const ob::Error& e) { if(probe) std::cout<<"D2C mode "<<mode<<": "<<e.getMessage()<<"\n"; }
          }
          if(colorProfile) break;
        }
        if(!colorProfile || !depthProfile) throw std::runtime_error("No matching D2C RGB/Y16 profiles. Refusing proportional depth mapping.");
        std::cout<<"Selected color "<<colorProfile->width()<<"x"<<colorProfile->height()<<" depth "<<depthProfile->width()<<"x"<<depthProfile->height()<<" @"<<colorProfile->fps()<<" D2C mode="<<alignMode<<"\n";
        if(probe) return 0;
        Detector detector(c);
        auto setup=std::make_shared<ob::Config>(); setup->enableStream(colorProfile); setup->enableStream(depthProfile);
        setup->setAlignMode(alignMode); setup->setDepthScaleRequire(true);
        setup->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_FULL_FRAME_REQUIRE);
        bool deviceSync=true;
        try { pipe.enableFrameSync(); }
        catch(const ob::Error& e) {
            deviceSync=false;
            std::cout<<"SDK frame sync unavailable: "<<e.getMessage()<<" Using host-receipt pairing for STATIONARY scenes only (not exposure synchronization).\n";
        }
        pipe.start(setup);
        struct StopPipe { ob::Pipeline& pipe; ~StopPipe(){try{pipe.stop();}catch(...){}} } stopPipe{pipe};
        auto params=pipe.getCameraParam();
        if(params.isMirrored) throw std::runtime_error("Mirrored camera calibration is unsupported; disable mirroring first");
        auto k=params.rgbIntrinsic; camera.size={k.width,k.height}; camera.K={k.fx,0,k.cx,0,k.fy,k.cy,0,0,1};
        auto distortion=params.rgbDistortion;
        camera.distortion=(cv::Mat_<double>(1,8)<<distortion.k1,distortion.k2,distortion.p1,distortion.p2,distortion.k3,distortion.k4,distortion.k5,distortion.k6);
        cv::Mat rays=makeRays(camera);
        Shared s; s.continuous=c.continuous!=0; Workers workers(s);
        workers.capture=std::thread([&] {
            try {
                auto lastFrame=Clock::now();
                std::deque<std::shared_ptr<ob::ColorFrame>> colorBuffer;
                uint64_t usedColorHost=0;
                while(!s.stop) {
                    auto frames=pipe.waitForFrames(200);
                    if(!frames) { if(elapsed(lastFrame)>5000) throw std::runtime_error("No complete RGB-D frames for 5 seconds"); continue; }
                    auto color=frames->colorFrame(); auto depth=frames->depthFrame();
                    if(!color || !depth) continue;
                    auto newestColor=color;
                    uint64_t newestColorHost=color->systemTimeStamp();
                    // Different device clock domains on DaBai DC1: pair by nearest host receipt.
                    // This is approximate and supports only stationary stop-and-look operation.
                    if(!deviceSync) {
                        if(colorBuffer.empty() || colorBuffer.back()->systemTimeStamp()!=color->systemTimeStamp()) colorBuffer.push_back(color);
                        while(colorBuffer.size()>8) colorBuffer.pop_front();
                        double best=1e99;
                        for(auto& candidate:colorBuffer) {
                            double difference=std::abs(double(candidate->systemTimeStamp())-double(depth->systemTimeStamp()));
                            if(candidate->systemTimeStamp()>usedColorHost && difference<best) { best=difference; color=candidate; }
                        }
                        if(color->systemTimeStamp()<=usedColorHost) continue;
                        if(best<=c.syncMs) usedColorHost=color->systemTimeStamp();
                    }
                    auto p=std::make_shared<Packet>();
                    auto bufferedMs=std::max<int64_t>(0,int64_t(newestColorHost)-int64_t(color->systemTimeStamp()));
                    p->received=Clock::now()-std::chrono::milliseconds(bufferedMs);
                    auto start=Clock::now(); p->color=decodeColor(color);
                    if(p->color.empty()) continue;
                    p->preview=newestColor==color?p->color:decodeColor(newestColor);
                    if(!supportedDepth(depth->format()) || cv::Size(depth->width(),depth->height())!=p->color.size() || p->color.size()!=camera.size)
                        throw std::runtime_error("D2C/intrinsic resolution mismatch; refusing invalid coordinates");
                    if(depth->dataSize()<size_t(depth->width())*depth->height()*2) throw std::runtime_error("Truncated depth frame");
                    float scale=depth->getValueScale(); if(!std::isfinite(scale) || scale<=0) throw std::runtime_error("Invalid depth scale");
                    cv::Mat(depth->height(),depth->width(),CV_16U,depth->data()).convertTo(p->depth,CV_32F,scale/1000.0);
                    p->deviceColorUs=double(color->timeStampUs()); p->deviceDepthUs=double(depth->timeStampUs());
                    p->hostReceivedMs=double(color->systemTimeStamp());
                    p->depthHostReceivedMs=double(depth->systemTimeStamp());
                    p->synchronized=p->deviceColorUs>0 && p->deviceDepthUs>0 && std::abs(p->deviceColorUs-p->deviceDepthUs)<=c.syncMs*1000;
                    if(!deviceSync) {
                        p->pairing="host_receipt_approximate_stationary_only";
                        p->synchronized=p->hostReceivedMs>0 && p->depthHostReceivedMs>0 &&
                            std::abs(p->hostReceivedMs-p->depthHostReceivedMs)<=c.syncMs;
                    }
                    if(!p->synchronized) ++s.unsynchronized;
                    p->decodeMs=elapsed(start); p->id=++s.captured;
                    { std::lock_guard<std::mutex> lock(s.mutex); s.latest=p; }
                    s.cv.notify_one(); lastFrame=Clock::now();
                }
            } catch(const ob::Error& e) { fail(s,e.getMessage()); }
              catch(const std::exception& e) { fail(s,e.what()); }
        });
        workers.inference=std::thread([&] {
            try {
                uint64_t previous=0; auto next=Clock::now();
                while(!s.stop) {
                    std::shared_ptr<Packet> p;
                    {
                        std::unique_lock<std::mutex> lock(s.mutex);
                        s.cv.wait(lock,[&]{return s.stop || (s.latest && s.latest->id!=previous && (s.continuous || s.requested));});
                        if(s.stop) break;
                        if(Clock::now()<next) { s.cv.wait_until(lock,next,[&]{return s.stop.load();}); if(s.stop) break; }
                        p=s.latest; previous=p->id; s.requested=false;
                    }
                    auto r=std::make_shared<Result>(analyze(p,detector,rays,c));
                    writeResult(c.output+"/latest.json",*r,detector.names,false);
                    { std::lock_guard<std::mutex> lock(s.mutex); s.result=r; }
                    ++s.processed; next=Clock::now()+std::chrono::milliseconds(c.interval);
                }
            } catch(const std::exception& e) { fail(s,e.what()); }
        });
        if(!headless) { cv::namedWindow("Live - latest frame",cv::WINDOW_AUTOSIZE); cv::namedWindow("Detection - matched RGB-D",cv::WINDOW_AUTOSIZE); }
        std::cout<<"Ready. SPACE pause/resume; d detect once; TAB select; s save result; c capture RGB-D; v depth overlay; q quit.\n";
        auto start=Clock::now(),lastReport=start;
        uint64_t displayed=0,lastCapture=0,lastProcessed=0,lastResult=0; int selected=-1; bool showDepth=false;
        while(!s.stop && !interrupted && (seconds<=0 || elapsed(start)<seconds*1000)) {
            std::shared_ptr<Packet> p; std::shared_ptr<Result> r;
            { std::lock_guard<std::mutex> lock(s.mutex); p=s.latest; r=s.result; }
            if(!headless) {
                if(p && p->id!=displayed) {
                    cv::Mat live=(p->preview.empty()?p->color:p->preview).clone();
                    cv::putText(live,std::string("LIVE | detection ")+(s.continuous?"ON":"PAUSED")+" | SPACE / d / TAB / s / c",{8,24},cv::FONT_HERSHEY_SIMPLEX,0.45,{0,255,255},1);
                    cv::imshow("Live - latest frame",live); displayed=p->id;
                }
                if(r) {
                    if(r->frame->id!=lastResult) { selected=-1; lastResult=r->frame->id; }
                    cv::imshow("Detection - matched RGB-D",drawResult(*r,detector.names,selected,&camera));
                }
                int key=cv::waitKey(10)&255;
                if(key=='q' || key==27) break;
                if(key==' ') s.continuous=!s.continuous.load();
                if(key=='d') { s.continuous=false; s.requested=true; }
                if(key=='v') {
                    showDepth=!showDepth;
                    if(showDepth) cv::namedWindow("Aligned depth overlay",cv::WINDOW_AUTOSIZE);
                    else cv::destroyWindow("Aligned depth overlay");
                }
                if(showDepth && p) cv::imshow("Aligned depth overlay",depthOverlay(*p,c));
                if(key==9 && r && !r->candidates.empty()) selected=(selected+1)%r->candidates.size();
                if(key=='c' && p) {
                    std::string stem=c.output+"/capture_"+std::to_string(uint64_t(p->hostReceivedMs))+"_"+std::to_string(p->id);
                    savePacket(stem,*p,camera); std::cout<<"Captured "<<stem<<".yml\n";
                }
                if(key=='s' && r) {
                        std::string stem=c.output+"/result_"+std::to_string(uint64_t(r->frame->hostReceivedMs))+"_"+std::to_string(r->frame->id);
                        savePacket(stem,*r->frame,camera); writeResult(stem+".json",*r,detector.names,false,selected);
                        cv::imwrite(stem+"_annotated.png",drawResult(*r,detector.names,selected,&camera)); std::cout<<"Saved "<<stem<<".json\n";
                }
                s.cv.notify_one();
                if(cv::getWindowProperty("Live - latest frame",cv::WND_PROP_VISIBLE)<1 || cv::getWindowProperty("Detection - matched RGB-D",cv::WND_PROP_VISIBLE)<1) break;
            } else std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if(elapsed(lastReport)>=2000) {
                double dt=elapsed(lastReport)/1000; uint64_t cap=s.captured.load(),processed=s.processed.load();
                std::cout<<std::fixed<<std::setprecision(1)<<"capture="<<(cap-lastCapture)/dt<<"fps detection="<<(processed-lastProcessed)/dt<<"Hz skew_rejected="<<s.unsynchronized.load();
                if(r) std::cout<<" infer="<<r->inferenceMs<<"ms result_age="<<elapsed(r->frame->received)<<"ms objects="<<r->candidates.size();
                std::cout<<std::endl; lastCapture=cap; lastProcessed=processed; lastReport=Clock::now();
            }
        }
        s.stop=true; s.cv.notify_all();
        if(workers.capture.joinable()) workers.capture.join();
        if(workers.inference.joinable()) workers.inference.join();
        if(s.result) {
            savePacket(c.output+"/last_session",*s.result->frame,camera);
            cv::imwrite(c.output+"/last_session_annotated.png",drawResult(*s.result,detector.names,-1,&camera));
            cv::imwrite(c.output+"/last_session_depth_overlay.png",depthOverlay(*s.result->frame,c));
        }
        if(!headless) cv::destroyAllWindows();
        if(!s.error.empty()) throw std::runtime_error(s.error);
        return 0;
    } catch(const ob::Error& e) { std::cerr<<"Orbbec: "<<e.getMessage()<<"\n"; return 1; }
      catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<"\n"; return 1; }
}
