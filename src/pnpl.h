#ifndef __PNPL_H__
#define __PNPL_H__
#include <opencv2/opencv.hpp>

/**\brief Project n Points and Lines with g2o */
void PnPL(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
          const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d,
          const cv::Mat& K,
          cv::Mat& R, cv::Mat& t);

#endif