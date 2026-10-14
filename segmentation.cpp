#include "segmentation.hpp"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <map>
#include <cmath>
#include <stdexcept>

namespace grasp {
struct Segmenter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"grasp_segment"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Config config;
    std::string inputName;
    std::vector<std::string> outputNames;
    int size=0;
    explicit Impl(const Config& c):config(c) {
        options.SetIntraOpNumThreads(c.threads); options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning","0");
        session.reset(new Ort::Session(env,c.onnx.c_str(),options));
        Ort::AllocatorWithDefaultOptions allocator;
        inputName=session->GetInputNameAllocated(0,allocator).get();
        auto shape=session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if(shape.size()!=4 || shape[0]!=1 || shape[1]!=3 || shape[2]!=shape[3] || shape[2]<=0)
            throw std::runtime_error("Segmenter requires fixed [1,3,S,S] ONNX input");
        size=int(shape[2]);
        if(size!=c.input) throw std::runtime_error("input_size must match the exported segmentation ONNX model");
        for(size_t i=0;i<session->GetOutputCount();++i) outputNames.emplace_back(session->GetOutputNameAllocated(i,allocator).get());
    }
};
Segmenter::Segmenter(const Config& config):impl(new Impl(config)) {
    std::ifstream f(config.segmentNames);
    if(!f) throw std::runtime_error("Cannot open segmentation class names: "+config.segmentNames);
    std::string name; while(std::getline(f,name)) { if(!name.empty() && name.back()=='\r') name.pop_back(); if(!name.empty()) names.push_back(name); }
    if(names.empty()) throw std::runtime_error("Empty segmentation names");
    for(auto& name:config.classes) if(std::find(names.begin(),names.end(),name)==names.end()) throw std::runtime_error("Unknown segmentation class: "+name);
}
Segmenter::~Segmenter()=default;
std::vector<Detection> Segmenter::run(const cv::Mat& image,Result& timing) {
    auto start=Clock::now(); auto& m=*impl; int size=m.size;
    double scale=std::min(double(size)/image.cols,double(size)/image.rows);
    int w=std::lround(image.cols*scale),h=std::lround(image.rows*scale),px=(size-w)/2,py=(size-h)/2;
    double sx=double(w)/image.cols,sy=double(h)/image.rows;
    cv::Mat resized,input; cv::resize(image,resized,{w,h});
    cv::copyMakeBorder(resized,input,py,size-h-py,px,size-w-px,cv::BORDER_CONSTANT,{114,114,114});
    cv::Mat blob=cv::dnn::blobFromImage(input,1.0/255,cv::Size(size,size),cv::Scalar(),true,false);
    std::array<int64_t,4> shape{1,3,size,size};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(memory,blob.ptr<float>(),blob.total(),shape.data(),shape.size());
    const char* inputName=m.inputName.c_str(); std::vector<const char*> outputNames;
    for(auto& s:m.outputNames) outputNames.push_back(s.c_str());
    timing.preprocessMs=elapsed(start); start=Clock::now();
    auto outputs=m.session->Run(Ort::RunOptions{nullptr},&inputName,&tensor,1,outputNames.data(),outputNames.size());
    timing.inferenceMs=elapsed(start);
    const Ort::Value *boxes=nullptr,*proto=nullptr;
    for(auto& output:outputs) {
        auto dims=output.GetTensorTypeAndShapeInfo().GetShape();
        if(dims.size()==3) boxes=&output;
        if(dims.size()==4) proto=&output;
    }
    if(!boxes || !proto) throw std::runtime_error("Expected segmentation boxes and mask prototype outputs");
    auto ps=proto->GetTensorTypeAndShapeInfo().GetShape(),bs=boxes->GetTensorTypeAndShapeInfo().GetShape();
    if(ps[0]!=1 || bs[0]!=1 || ps[1]<=0 || ps[2]<=0 || ps[3]<=0) throw std::runtime_error("Unsupported segmentation output shape");
    int nm=int(ps[1]),mh=int(ps[2]),mw=int(ps[3]),nc=int(names.size());
    bool endToEnd=bs[2]==6+nm;
    if(!endToEnd && bs[1]!=4+nc+nm) throw std::runtime_error("ONNX segmentation head/names mismatch");
    int count=int(endToEnd?bs[1]:bs[2]);
    const float* b=boxes->GetTensorData<float>();
    cv::Mat prototypes(nm,mh*mw,CV_32F,const_cast<float*>(proto->GetTensorData<float>()));
    struct Raw { Detection det; std::vector<float> coefficients; };
    std::vector<Raw> raw;
    for(int i=0;i<count;++i) {
        auto at=[&](int channel){return endToEnd?b[i*int(bs[2])+channel]:b[channel*count+i];};
        int cls=0; float confidence=0;
        if(endToEnd) { confidence=at(4); if(!std::isfinite(at(5))) continue; cls=int(at(5)); }
        else for(int j=0;j<nc;++j) if(at(4+j)>confidence) { confidence=at(4+j); cls=j; }
        if(cls<0 || cls>=nc || !std::isfinite(confidence) || confidence<m.config.confidence) continue;
        if(!m.config.classes.empty() && std::find(m.config.classes.begin(),m.config.classes.end(),names[cls])==m.config.classes.end()) continue;
        double x1=at(0),y1=at(1),x2=at(2),y2=at(3);
        if(!endToEnd) { x1=at(0)-at(2)/2; y1=at(1)-at(3)/2; x2=at(0)+at(2)/2; y2=at(1)+at(3)/2; }
        if(!std::isfinite(x1+y1+x2+y2)) continue;
        int left=int(std::max(0.0,std::min(double(image.cols),std::floor((x1-px)/sx))));
        int top=int(std::max(0.0,std::min(double(image.rows),std::floor((y1-py)/sy))));
        int right=int(std::max(0.0,std::min(double(image.cols),std::ceil((x2-px)/sx))));
        int bottom=int(std::max(0.0,std::min(double(image.rows),std::ceil((y2-py)/sy))));
        if(right<=left || bottom<=top) continue;
        Raw detection; detection.det={cls,confidence,{left,top,right-left,bottom-top}};
        for(int j=0;j<nm;++j) detection.coefficients.push_back(at((endToEnd?6:4+nc)+j));
        raw.push_back(std::move(detection));
    }
    std::vector<int> keep;
    if(endToEnd) { for(size_t i=0;i<raw.size();++i) keep.push_back(int(i)); }
    else {
        std::map<int,std::vector<int>> groups;
        for(size_t i=0;i<raw.size();++i) groups[raw[i].det.classId].push_back(int(i));
        for(auto& group:groups) {
            std::vector<cv::Rect> rects; std::vector<float> scores;
            for(int i:group.second) { rects.push_back(raw[i].det.box); scores.push_back(raw[i].det.score); }
            std::vector<int> indices; cv::dnn::NMSBoxes(rects,scores,m.config.confidence,m.config.nms,indices);
            for(int index:indices) keep.push_back(group.second[index]);
        }
    }
    std::sort(keep.begin(),keep.end(),[&](int a,int b){return raw[a].det.score>raw[b].det.score;});
    if(keep.size()>32) keep.resize(32);
    std::vector<Detection> result;
    for(int index:keep) {
        auto& r=raw[index]; cv::Mat weights(1,nm,CV_32F,r.coefficients.data());
        cv::Mat logits=weights*prototypes; cv::Mat full,original;
        cv::resize(logits.reshape(1,mh),full,{size,size});
        cv::resize(full(cv::Rect(px,py,w,h)),original,image.size());
        r.det.mask=cv::Mat(image.size(),CV_8U,cv::Scalar(0));
        cv::Mat binary=original(r.det.box)>0; binary.copyTo(r.det.mask(r.det.box));
        result.push_back(std::move(r.det));
    }
    return result;
}
}
