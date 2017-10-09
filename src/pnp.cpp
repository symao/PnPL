#include "pnp.h"

void p2pTrans(const cv::Point3f& pt1, const cv::Point2f& uv1,
              const cv::Point3f& pt2, const cv::Point2f& uv2, cv::Point3f& trans)
{
    float a[9] = {0};
    float b[3] = {0};
    a[0] = a[4] = 2;
    a[2] = a[6] = -uv1.x-uv2.x;
    a[5] = a[7] = -uv1.y-uv2.y;
    a[8] = (uv1.x*uv1.x + uv1.y*uv1.y) + (uv2.x*uv2.x + uv2.y*uv2.y);
    b[0] = (uv1.x*pt1.z - pt1.x) + (uv2.x*pt2.z - pt2.x);
    b[1] = (uv1.y*pt1.z - pt1.y) + (uv2.y*pt2.z - pt2.y);
    b[2] = (uv1.x*(pt1.x - uv1.x*pt1.z) + uv1.y*(pt1.y - uv1.y*pt1.z)) + 
           (uv2.x*(pt2.x - uv2.x*pt2.z) + uv2.y*(pt2.y - uv2.y*pt2.z));
    cv::Mat t = cv::Mat(3,3,CV_32FC1,a).inv()*cv::Mat(3,1,CV_32FC1,b);
    trans = cv::Point3f(t);
}

void pnpTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                cv::Point3f& trans)
{
    if(pts3d.size()<2) {return;}
    float a[9] = {0};
    float b[3] = {0};
    int N = pts3d.size();
    for(int i=0; i<N; i++)
    {
        const auto& pti = pts3d[i];
        const auto& uvi = pts2d[i];
        a[2] -= uvi.x;
        a[5] -= uvi.y;
        a[8] += (uvi.x*uvi.x + uvi.y*uvi.y);
        float b0 = uvi.x*pti.z - pti.x;
        float b1 = uvi.y*pti.z - pti.y;
        b[0] += b0;
        b[1] += b1;
        b[2] += -uvi.x*b0 - uvi.y*b1;
    }
    a[0] = a[4] = N;
    a[6] = a[2];
    a[7] = a[5];
    cv::Mat t = cv::Mat(3,3,CV_32FC1,a).inv()*cv::Mat(3,1,CV_32FC1,b);
    trans = cv::Point3f(t);
}

void pnlTrans(const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d,
                cv::Point3f& trans)
{
    int N = lns3d.size();
    if(N<3) return;
    float a[9] = {0};
    float b[3] = {0};
    for(int i=0; i<N; i++)
    {
        const auto ln3d = lns3d[i];
        const auto ln2d = lns2d[i];

        float u1 = ln2d[0];
        float v1 = ln2d[1];
        float u2 = ln2d[2];
        float v2 = ln2d[3];
        float n[3] = {v1-v2, -u1+u2, u1*v2-u2*v1};
        if(n[2]==0) continue;
        n[0]/=n[2];
        n[1]/=n[2];
        n[2]=1;

        float ai[9] = { n[0]*n[0],n[0]*n[1],n[0]*n[2],
                        n[1]*n[0],n[1]*n[1],n[1]*n[2],
                        n[2]*n[0],n[2]*n[1],n[2]*n[2]};
        float x = (ln3d[0]+ln3d[3])/2.0;
        float y = (ln3d[1]+ln3d[4])/2.0;
        float z = (ln3d[2]+ln3d[5])/2.0;
        for(int j=0;j<9;j++) {a[j]+=ai[j];}

        b[0] -= ai[0]*x + ai[1]*y + ai[2]*z;
        b[1] -= ai[3]*x + ai[4]*y + ai[5]*z;
        b[2] -= ai[6]*x + ai[7]*y + ai[8]*z;
    }
    cv::Mat t = cv::Mat(3,3,CV_32FC1,a).inv()*cv::Mat(3,1,CV_32FC1,b);
    trans = cv::Point3f(t);
}

void pnplTrans(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
                const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d,
                cv::Point3f& trans)
{
    int Np = pts3d.size();
    int Nl = lns3d.size();
    if(Np<2 || Nl<3) {return;}

    float a[9] = {0};
    float b[3] = {0};
    // pnp
    for(int i=0; i<Np; i++)
    {
        const auto& pti = pts3d[i];
        const auto& uvi = pts2d[i];
        a[2] -= uvi.x;
        a[5] -= uvi.y;
        a[8] += (uvi.x*uvi.x + uvi.y*uvi.y);
        float b0 = uvi.x*pti.z - pti.x;
        float b1 = uvi.y*pti.z - pti.y;
        b[0] += b0;
        b[1] += b1;
        b[2] += -uvi.x*b0 - uvi.y*b1;
    }
    a[0] = a[4] = Np;
    a[6] = a[2];
    a[7] = a[5];
    // pnl
    for(int i=0; i<Nl; i++)
    {
        const auto ln3d = lns3d[i];
        const auto ln2d = lns2d[i];

        float u1 = ln2d[0];
        float v1 = ln2d[1];
        float u2 = ln2d[2];
        float v2 = ln2d[3];
        float n[3] = {v1-v2, -u1+u2, u1*v2-u2*v1};
        if(n[2]==0) continue;
        n[0]/=n[2];
        n[1]/=n[2];
        n[2]=1;

        float ai[9] = { n[0]*n[0],n[0]*n[1],n[0]*n[2],
                        n[1]*n[0],n[1]*n[1],n[1]*n[2],
                        n[2]*n[0],n[2]*n[1],n[2]*n[2]};
        float x = (ln3d[0]+ln3d[3])/2.0;
        float y = (ln3d[1]+ln3d[4])/2.0;
        float z = (ln3d[2]+ln3d[5])/2.0;
        for(int j=0;j<9;j++) {a[j]+=ai[j];}

        b[0] -= ai[0]*x + ai[1]*y + ai[2]*z;
        b[1] -= ai[3]*x + ai[4]*y + ai[5]*z;
        b[2] -= ai[6]*x + ai[7]*y + ai[8]*z;
    }
    cv::Mat t = cv::Mat(3,3,CV_32FC1,a).inv()*cv::Mat(3,1,CV_32FC1,b);
    trans = cv::Point3f(t);
}