#ifndef __VS_PNP_H__
#define __VS_PNP_H__
#include <opencv2/opencv.hpp>

// project 2 points, assert K in ones, so 2d points are projected into normalized plane
void p2pTrans(const cv::Point3f& pt1, const cv::Point2f& uv1,
              const cv::Point3f& pt2, const cv::Point2f& uv2,
              cv::Point3f& trans);

// project n points, need at least 2 points, assert K in ones, so 2d points are projected into normalized plane
void pnpTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                cv::Point3f& trans);


// project n lines, need at least 3 lines, assert K in ones, so 2d lines are projected into normalized plane
void pnlTrans(const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d,
                cv::Point3f& trans);

// project n points and lines, need at least 3 lines, or at least 2 points
void pnplTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d,
                cv::Point3f& trans);

// projection su = Rz(yaw)p + t, where u in pts2d and p in pts3d
bool pnpRansacTransYaw(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                    cv::Point3f& trans, float& yaw, std::vector<int>& inlier_idx, int ite_cnt = 100, float reprj_err = 8.0);

#endif