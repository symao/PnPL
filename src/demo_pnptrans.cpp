#include "pnp.h"

int main(int argc, char* argv[])
{
    // build data
    cv::Point3f t(1,2,3);
    std::vector<cv::Point3f> pts3d = {cv::Point3f(10,1,5),cv::Point3f(3,1,5),cv::Point3f(10,2,5),cv::Point3f(3,2,5)};
    std::vector<cv::Point2f> pts2d;
    for(auto pt: pts3d)
    {
        auto pt_new = pt+t;
        pts2d.push_back(cv::Point2f(pt_new.x/pt_new.z,pt_new.y/pt_new.z));
    }
    cv::Point3f trans;
    std::vector<cv::Vec6f> lns3d = {cv::Vec6f(pts3d[0].x,pts3d[0].y,pts3d[0].z,pts3d[1].x,pts3d[1].y,pts3d[1].z),
                                    cv::Vec6f(pts3d[0].x,pts3d[0].y,pts3d[0].z,pts3d[2].x,pts3d[2].y,pts3d[2].z),
                                    cv::Vec6f(pts3d[2].x,pts3d[2].y,pts3d[2].z,pts3d[3].x,pts3d[3].y,pts3d[3].z)};
    std::vector<cv::Vec4f> lns2d = {cv::Vec4f(pts2d[0].x,pts2d[0].y,pts2d[1].x,pts2d[1].y),
                                    cv::Vec4f(pts2d[0].x,pts2d[0].y,pts2d[2].x,pts2d[2].y),
                                    cv::Vec4f(pts2d[2].x,pts2d[2].y,pts2d[3].x,pts2d[3].y)};
    for(auto& l:lns2d)
    {
        float dx = l[2]-l[0];
        float dy = l[3]-l[1];
        l[0]+=dx*100.1;
        l[1]+=dy*100.1;
        l[2]+=dx*200.4;
        l[3]+=dy*200.4;
    }

    // test pnp trans
    pnplTrans(pts3d,pts2d,lns3d,lns2d,trans);
    std::cout<<trans<<std::endl;
    pnlTrans(lns3d,lns2d,trans);
    std::cout<<trans<<std::endl;
}