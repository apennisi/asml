/*
 *  ASML - Skin Lesion Image Segmentation Using Delaunay Triangulation for
 *  Melanoma Detection
 *
 *
 *  Written by Andrea Pennisi
 *
 *  Please, report suggestions/comments/bugs to
 *  andrea.pennisi@gmail.com
 *
 */

#ifndef SKINDETECTION_H
#define SKINDETECTION_H

#include <opencv2/opencv.hpp>

class SkinDetection
{
public:
    SkinDetection(const cv::Mat &img);
    inline void setParameters(const int &Y, const int &Cr, const int &Cb,
                              const int &H, const int &s, const int &v);
    void compute();
    inline cv::Mat getImg() const { return neo; }
    inline cv::Mat getBw() const { return bw; }
    inline double getPerc() const { return perc; }
private:
    int Y_MIN, Y_MAX, Cr_MIN, Cr_MAX, Cb_MIN, Cb_MAX,
        Hmin, Hmax, smin, smax, vmin, vmax;
    cv::Mat img, hsv, ycrcb, neo;
    cv::Mat bw;
    double perc;
};

#endif // SKINDETECTION_H
