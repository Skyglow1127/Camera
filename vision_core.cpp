#include "vision_core.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

namespace grasp {
void Camera::validate() const {
    if(size.width <= 0 || size.height <= 0 || !cv::checkRange(cv::Mat(K)) ||
       K(0,0) <= 0 || K(1,1) <= 0 || std::abs(K(2,2)-1) > 1e-8 ||
       (!distortion.empty() && !cv::checkRange(distortion)))
        throw std::runtime_error("Invalid camera intrinsics");
}
cv::Mat makeRays(const Camera& camera) {
    camera.validate();
    std::vector<cv::Point2f> pixels, normalized;
    pixels.reserve(camera.size.area());
    for(int y=0; y<camera.size.height; ++y)
        for(int x=0; x<camera.size.width; ++x) pixels.emplace_back(x,y);
    cv::undistortPoints(pixels, normalized, camera.K, camera.distortion);
    cv::Mat rays(camera.size, CV_32FC2);
    std::copy(normalized.begin(), normalized.end(), rays.ptr<cv::Point2f>());
    return rays;
}
cv::Vec3d pointAt(const cv::Mat& rays, int x, int y, double z) {
    auto r = rays.at<cv::Vec2f>(y,x);
    return {r[0]*z, r[1]*z, z};
}
double percentile(std::vector<double> v, double q) {
    if(v.empty()) throw std::runtime_error("Empty percentile input");
    size_t i = static_cast<size_t>(std::max(0.0,std::min(1.0,q))*(v.size()-1));
    std::nth_element(v.begin(), v.begin()+i, v.end());
    return v[i];
}
static bool validZ(float z, const Settings& s) {
    return std::isfinite(z) && z >= s.minDepth && z <= s.maxDepth;
}
static void validateDepth(const cv::Mat& depth, const cv::Mat& rays) {
    if(depth.type()!=CV_32FC1 || rays.type()!=CV_32FC2 || depth.size()!=rays.size())
        throw std::runtime_error("Depth must be aligned CV_32F metres with matching ray map");
}
Plane fitTable(const cv::Mat& depth, const cv::Mat& rays,
               const std::vector<Detection>& detections, const Settings& s) {
    validateDepth(depth,rays);
    cv::Mat excluded(depth.size(),CV_8U,cv::Scalar(0));
    for(auto& d:detections) {
        cv::Rect box=d.box & cv::Rect(0,0,depth.cols,depth.rows);
        if(!d.mask.empty()) {
            if(d.mask.size()!=depth.size() || d.mask.type()!=CV_8U) throw std::runtime_error("Invalid instance mask");
            cv::bitwise_or(excluded,d.mask,excluded);
        } else if(box.area()>0) excluded(box).setTo(255);
    }
    std::vector<cv::Vec3d> points;
    int step=std::max(2,static_cast<int>(std::sqrt(depth.total()/3500.0)));
    for(int y=0;y<depth.rows;y+=step) for(int x=0;x<depth.cols;x+=step) {
        float z=depth.at<float>(y,x);
        if(!excluded.at<uchar>(y,x) && validZ(z,s)) points.push_back(pointAt(rays,x,y,z));
    }
    Plane result;
    if(points.size()<80) return result;
    std::mt19937 rng(73);
    std::uniform_int_distribution<size_t> pick(0,points.size()-1);
    size_t best=0;
    for(int i=0;i<160;++i) {
        cv::Vec3d a=points[pick(rng)], b=points[pick(rng)], c=points[pick(rng)];
        cv::Vec3d n=(b-a).cross(c-a);
        double length=cv::norm(n);
        if(length<1e-9) continue;
        n/=length;
        double offset=-n.dot(a);
        if(offset<0) { n=-n; offset=-offset; }
        // Only support a table viewed reasonably from above in this first version.
        if(n[2]>-0.5 || offset<0.05) continue;
        size_t count=0;
        for(auto& p:points) if(std::abs(n.dot(p)+offset)<s.planeThreshold) ++count;
        if(count>best) { best=count; result.normal=n; result.offset=offset; }
    }
    result.inlierRatio=static_cast<double>(best)/points.size();
    if(best<80 || result.inlierRatio<s.minPlaneRatio) return result;
    cv::Mat samples(static_cast<int>(best),3,CV_64F);
    int row=0;
    for(auto& p:points) if(std::abs(result.normal.dot(p)+result.offset)<s.planeThreshold) {
        for(int k=0;k<3;++k) samples.at<double>(row,k)=p[k];
        ++row;
    }
    cv::PCA pca(samples,cv::Mat(),cv::PCA::DATA_AS_ROW);
    cv::Vec3d n(pca.eigenvectors.at<double>(2,0),pca.eigenvectors.at<double>(2,1),pca.eigenvectors.at<double>(2,2));
    cv::Vec3d mean(pca.mean.at<double>(0,0),pca.mean.at<double>(0,1),pca.mean.at<double>(0,2));
    double d=-n.dot(mean);
    if(d<0) { n=-n; d=-d; }
    double error=0; size_t count=0;
    for(auto& p:points) {
        double e=n.dot(p)+d;
        if(std::abs(e)<s.planeThreshold) { error+=e*e; ++count; }
    }
    result.normal=n; result.offset=d;
    result.inlierRatio=static_cast<double>(count)/points.size();
    result.rms=count ? std::sqrt(error/count) : 1;
    result.valid=count>=80 && result.inlierRatio>=s.minPlaneRatio && n[2]<-0.5;
    return result;
}
std::vector<Detection> tableObjects(const cv::Mat& depth,const cv::Mat& rays,const Plane& plane,const Settings& s) {
    validateDepth(depth,rays);
    std::vector<Detection> result;
    if(!plane.valid) return result;
    cv::Mat foreground(depth.size(),CV_8U,cv::Scalar(0));
    for(int y=0;y<depth.rows;++y) for(int x=0;x<depth.cols;++x) {
        float z=depth.at<float>(y,x);
        if(!validZ(z,s)) continue;
        double h=plane.normal.dot(pointAt(rays,x,y,z))+plane.offset;
        if(h>=s.minHeight && h<=s.maxHeight) foreground.at<uchar>(y,x)=255;
    }
    cv::Mat labels,stats,centroids;
    int count=cv::connectedComponentsWithStats(foreground,labels,stats,centroids,8);
    for(int i=1;i<count;++i) {
        if(stats.at<int>(i,cv::CC_STAT_AREA)<s.minObjectPixels) continue;
        cv::Rect box(stats.at<int>(i,cv::CC_STAT_LEFT),stats.at<int>(i,cv::CC_STAT_TOP),
                     stats.at<int>(i,cv::CC_STAT_WIDTH),stats.at<int>(i,cv::CC_STAT_HEIGHT));
        // Components cut by the image boundary are incomplete objects.
        if(box.x<=0 || box.y<=0 || box.br().x>=depth.cols || box.br().y>=depth.rows) continue;
        result.push_back({0,1.f,box,labels==i}); // Geometry confidence is not a learned class probability.
    }
    std::sort(result.begin(),result.end(),[](const Detection&a,const Detection&b){return a.box.area()>b.box.area();});
    if(result.size()>32) result.resize(32);
    return result;
}
Candidate estimate(const Detection& det, const cv::Mat& depth,
                   const cv::Mat& rays, const Plane& plane, const Settings& s) {
    validateDepth(depth,rays);
    Candidate c; c.detection=det;
    cv::Rect roi=det.box & cv::Rect(0,0,depth.cols,depth.rows);
    if(roi.area()==0) { c.reason="empty_box"; return c; }
    if(!det.mask.empty() && (det.mask.type()!=CV_8U || det.mask.size()!=depth.size())) throw std::runtime_error("Invalid instance mask");
    // Depth observation is useful even when no trustworthy tabletop/grasp exists.
    cv::Mat sampling(roi.size(),CV_8U,cv::Scalar(0));
    if(!det.mask.empty()) cv::erode(det.mask(roi),sampling,cv::Mat(),{-1,-1},1,cv::BORDER_CONSTANT,cv::Scalar(0));
    else {
        cv::Rect center(roi.width/4,roi.height/4,std::max(1,roi.width/2),std::max(1,roi.height/2));
        sampling(center & cv::Rect(0,0,roi.width,roi.height)).setTo(255);
    }
    std::vector<double> observations;
    for(int y=0;y<roi.height;++y) for(int x=0;x<roi.width;++x)
        if(sampling.at<uchar>(y,x) && validZ(depth.at<float>(y+roi.y,x+roi.x),s)) observations.push_back(depth.at<float>(y+roi.y,x+roi.x));
    int samplingPixels=cv::countNonZero(sampling);
    c.validRatio=samplingPixels?double(observations.size())/samplingPixels:0;
    if(observations.size()>=5 && c.validRatio>=s.minValidRatio) {
        double z=percentile(observations,0.5),best=1e99;
        for(int y=0;y<roi.height;++y) for(int x=0;x<roi.width;++x) {
            if(!sampling.at<uchar>(y,x)) continue;
            float value=depth.at<float>(y+roi.y,x+roi.x);
            if(!validZ(value,s) || std::abs(value-z)>s.planeThreshold) continue;
            double distance=std::pow(x-roi.width/2.0,2)+std::pow(y-roi.height/2.0,2);
            if(distance<best) { best=distance; c.pixel={x+roi.x,y+roi.y}; }
        }
        if(c.pixel.x>=0) { c.surface=pointAt(rays,c.pixel.x,c.pixel.y,z); c.measured=true; }
    }
    if(!plane.valid) { c.reason="table_not_found"; return c; }
    cv::Mat foreground(roi.size(),CV_8U,cv::Scalar(0));
    int valid=0;
    for(int y=0;y<roi.height;++y) for(int x=0;x<roi.width;++x) {
        if(!det.mask.empty() && !det.mask.at<uchar>(y+roi.y,x+roi.x)) continue;
        float z=depth.at<float>(y+roi.y,x+roi.x);
        if(!validZ(z,s)) continue;
        ++valid;
        double h=plane.normal.dot(pointAt(rays,x+roi.x,y+roi.y,z))+plane.offset;
        if(h>=s.minHeight && h<=s.maxHeight) foreground.at<uchar>(y,x)=255;
    }
    int denominator=det.mask.empty()?roi.area():cv::countNonZero(det.mask(roi));
    c.validRatio=denominator?static_cast<double>(valid)/denominator:0;
    if(c.validRatio<s.minValidRatio) { c.reason="insufficient_depth"; return c; }
    cv::Mat labels,stats,centroids;
    // No hole filling: an unmeasured hole must not become a measured grasp point.
    int count=cv::connectedComponentsWithStats(foreground,labels,stats,centroids,8);
    int largest=0, area=0;
    for(int i=1;i<count;++i) if(stats.at<int>(i,cv::CC_STAT_AREA)>area) {
        largest=i; area=stats.at<int>(i,cv::CC_STAT_AREA);
    }
    if(area<s.minObjectPixels) { c.reason="no_object_above_table"; return c; }
    cv::Mat component=(labels==largest), interior;
    cv::erode(component,interior,cv::Mat(),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
    if(cv::countNonZero(interior)<s.minObjectPixels) { c.reason="too_few_interior_pixels"; return c; }
    c.mask=cv::Mat(depth.size(),CV_8U,cv::Scalar(0));
    component.copyTo(c.mask(roi));
    cv::Vec3d e1=cv::Vec3d(1,0,0)-plane.normal*plane.normal[0]; e1/=cv::norm(e1);
    cv::Vec3d e2=plane.normal.cross(e1);
    std::vector<cv::Vec3d> points;
    std::vector<cv::Point> pixels;
    for(int y=0;y<roi.height;++y) for(int x=0;x<roi.width;++x) if(component.at<uchar>(y,x)) {
        pixels.emplace_back(x+roi.x,y+roi.y);
        points.push_back(pointAt(rays,x+roi.x,y+roi.y,depth.at<float>(y+roi.y,x+roi.x)));
    }
    cv::Mat footprint(static_cast<int>(points.size()),2,CV_64F);
    for(size_t i=0;i<points.size();++i) {
        footprint.at<double>(i,0)=e1.dot(points[i]);
        footprint.at<double>(i,1)=e2.dot(points[i]);
    }
    cv::PCA pca(footprint,cv::Mat(),cv::PCA::DATA_AS_ROW);
    cv::Vec3d closing=e1*pca.eigenvectors.at<double>(1,0)+e2*pca.eigenvectors.at<double>(1,1);
    // Resolve the sign ambiguity for reproducible output; symmetric objects still have ambiguous yaw.
    if(closing.dot(e1)<0) closing=-closing;
    cv::Vec3d approach=-plane.normal, side=approach.cross(closing);
    for(int k=0;k<3;++k) { c.axes(k,0)=closing[k]; c.axes(k,1)=side[k]; c.axes(k,2)=approach[k]; }
    c.orientationRatio=pca.eigenvalues.at<double>(0)/std::max(1e-12,pca.eigenvalues.at<double>(1));
    std::vector<double> widths,heights;
    for(auto& p:points) { widths.push_back(closing.dot(p)); heights.push_back(plane.normal.dot(p)+plane.offset); }
    c.opening=percentile(widths,0.98)-percentile(widths,0.02)+2*s.fingerMargin;
    c.height=percentile(heights,0.5);
    double best=1e99;
    cv::Point graspPixel(-1,-1);
    for(size_t i=0;i<points.size();++i) {
        auto px=pixels[i];
        if(!interior.at<uchar>(px.y-roi.y,px.x-roi.x)) continue;
        double a=e1.dot(points[i])-pca.mean.at<double>(0,0);
        double b=e2.dot(points[i])-pca.mean.at<double>(0,1);
        double score=a*a+b*b;
        if(score<best) { best=score; graspPixel=px; }
    }
    std::vector<double> localDepth;
    for(int y=std::max(0,graspPixel.y-2);y<=std::min(depth.rows-1,graspPixel.y+2);++y)
        for(int x=std::max(0,graspPixel.x-2);x<=std::min(depth.cols-1,graspPixel.x+2);++x)
            if(c.mask.at<uchar>(y,x)) localDepth.push_back(depth.at<float>(y,x));
    if(localDepth.size()<5) { c.reason="insufficient_local_depth"; return c; }
    double z=percentile(localDepth,0.5);
    std::vector<double> deviations;
    for(double v:localDepth) deviations.push_back(std::abs(v-z));
    c.depthMad=percentile(deviations,0.5);
    c.pixel=graspPixel; c.surface=pointAt(rays,c.pixel.x,c.pixel.y,z);
    c.measured=true;
    if(c.depthMad>s.planeThreshold) { c.reason="unstable_surface_depth"; return c; }
    if(c.opening>s.maxOpening) { c.reason="exceeds_gripper_opening"; return c; }
    c.valid=true; c.reason="geometric_candidate_only";
    return c;
}
std::vector<Detection> classNms(const std::vector<Detection>& input,float score,float iou) {
    std::map<int,std::vector<Detection>> groups;
    for(auto& d:input) if(d.box.area()>0 && std::isfinite(d.score)) groups[d.classId].push_back(d);
    std::vector<Detection> result;
    for(auto& item:groups) {
        std::vector<cv::Rect> boxes; std::vector<float> scores;
        for(auto& d:item.second) { boxes.push_back(d.box); scores.push_back(d.score); }
        std::vector<int> keep; cv::dnn::NMSBoxes(boxes,scores,score,iou,keep);
        for(int i:keep) result.push_back(item.second[i]);
    }
    std::sort(result.begin(),result.end(),[](const Detection&a,const Detection&b){return a.score>b.score;});
    return result;
}
bool validTransform(const cv::Mat& input) {
    if(input.rows!=4 || input.cols!=4 || input.channels()!=1 || !cv::checkRange(input)) return false;
    cv::Mat m; input.convertTo(m,CV_64F);
    cv::Mat R=m(cv::Rect(0,0,3,3));
    cv::Mat bottom=(cv::Mat_<double>(1,4)<<0,0,0,1);
    return cv::norm(m.row(3)-bottom)<1e-6 && cv::norm(R.t()*R-cv::Mat::eye(3,3,CV_64F))<1e-3 &&
           std::abs(cv::determinant(R)-1)<1e-3;
}
cv::Matx44d asTransform(const cv::Mat& m) {
    if(!validTransform(m)) throw std::runtime_error("Invalid rigid transform (expected 4x4, metres, proper rotation)");
    cv::Mat d; m.convertTo(d,CV_64F); cv::Matx44d result;
    std::copy(d.ptr<double>(),d.ptr<double>()+16,result.val); return result;
}
cv::Matx44d surfaceTransform(const Candidate& c) {
    cv::Matx44d t=cv::Matx44d::eye();
    for(int i=0;i<3;++i) { t(i,3)=c.surface[i]; for(int j=0;j<3;++j) t(i,j)=c.axes(i,j); }
    return t;
}
}
