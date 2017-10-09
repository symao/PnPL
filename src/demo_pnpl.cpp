#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "pnpl.h"

void build_data(std::vector<cv::Point3f>& pts3d, std::vector<cv::Point2f>& pts2d,
                std::vector<cv::Vec6f>& lns3d, std::vector<cv::Vec4f>& lns2d,
                cv::Mat& K)
{
    float noise_pixel = 1;
    auto get_rand = [noise_pixel](){return noise_pixel*((rand()%1001-500)/500.0);};

    double f = 800;
    double cx = 320;
    double cy = 240;
    K = (cv::Mat_<double>(3,3)<<f,0,cx,0,f,cy,0,0,1);
    pts3d = {cv::Point3f(0,0,0),cv::Point3f(0.5,0.5,0),cv::Point3f(0.5,-0.5,0),cv::Point3f(-0.5,-0.5,0),cv::Point3f(-0.5,0.5,0)};

    Eigen::Matrix3d R = (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())*
                        Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY())*
                        Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX())
                        ).matrix();
    Eigen::Vector3d t(1,-2,10);

    pts2d.clear();
    for(const auto & pt: pts3d)
    {
        Eigen::Vector3d pt_new = R*Eigen::Vector3d(pt.x,pt.y,pt.z) + t;
        pts2d.push_back(cv::Point2f(f*pt_new(0)/pt_new(2)+cx, f*pt_new(1)/pt_new(2)+cy));
    }

    lns3d = {cv::Vec6f(0.3,0.4,0,0.5,-0.5,0),cv::Vec6f(0.5,0.5,0,-0.5,0.5,0),cv::Vec6f(0,0,0,-0.5,-0.5,0)};
    lns2d.clear();
    for(const auto& ln: lns3d)
    {
        Eigen::Vector3d pt1, pt2;
        pt1<<ln[0],ln[1],ln[2];
        pt2<<ln[3],ln[4],ln[5];
        pt1 = R*pt1+t;
        pt2 = R*pt2+t;
        lns2d.push_back(cv::Vec4f(f*pt1(0)/pt1(2)+cx, f*pt1(1)/pt1(2)+cy, f*pt2(0)/pt2(2)+cx, f*pt2(1)/pt2(2)+cy));
    }

    for(auto& pt:pts2d)
    {
        pt.x+=get_rand();
        pt.y+=get_rand();
    }
    for(auto& ln:lns2d)
    {
        float dx = ln[2]-ln[0];
        float dy = ln[3]-ln[1];
        float n = hypotf(dx,dy);
        dx/=n;
        dy/=n;
        ln[0]+=dx*0.1;
        ln[1]+=dy*0.1;
        ln[2]-=dx-0.2;
        ln[3]-=dy-0.2;

        ln[0]+=get_rand();
        ln[1]+=get_rand();
        ln[2]+=get_rand();
        ln[3]+=get_rand();
    }
}


int main()
{
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<cv::Vec6f> lns3d;
    std::vector<cv::Vec4f> lns2d;
    cv::Mat K;
    build_data(pts3d,pts2d,lns3d,lns2d,K);

    // for(const auto p:pts2d)
    //     std::cout<<p<<" ";
    // std::cout<<std::endl;
    // for(const auto l:lns2d)
    //     std::cout<<l<<" ";
    // std::cout<<std::endl;
    // std::cout<<"K:"<<K<<std::endl;

    cv::Mat R,t;
    PnPL(pts3d,pts2d,lns3d,lns2d,K,R,t);
    std::cout<<R<<std::endl;
    std::cout<<t<<std::endl;
}
