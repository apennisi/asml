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

#ifndef RGBSEGMENTATION_H
#define RGBSEGMENTATION_H


#include <opencv2/opencv.hpp>
#include <delaunaytri.h>
#include <rasterizetriangles.h>
#include <mergetriangles.h>
#include <stdio.h>
#include <math.h>
#include <utils.h>
#include <fstream>
#include <sstream>
#include <time.h>

class AccumElem {
public:
    AccumElem() {
        index = 0;
        sum = 0.;
    }

    int index;
    double sum;
};


class RgbSegmentation {

    public:
        RgbSegmentation(const cv::Mat &img, const cv::Mat &rgbImg,
                        std::vector<cv::Point2i> XYPoints,
                        const double &sigma_blurring, const double &sigma_c,
                        const double &merge_threshold, cv::Mat &RgbLabels);
        virtual ~RgbSegmentation();
        inline cv::Mat getMat() const {return imgSeg;}
        inline cv::Mat getEdges() const {return edges; }
    private:
        struct compare_xy {
            bool operator ()(const cv::Point2i& left, const cv::Point2i& right) const
            {
                return (left.x == right.x ? left.y < right.y : left.x < right.x);
            }
        };
    private:
        cv::Mat imgSeg;
        int rows, cols, pixelNumber, sizeTriangles;
        std::vector<double> gaussKernel;
        std::vector<double> overlap_areas, overlap_ratios, edge_lengths, mean_color_rgb, mean_color_lab;
        std::vector<double> color_weights;
        std::vector<double> neighbors;
        std::vector<double> merge_costs, merge_lengths;
        std::vector<int> new_labels, old_labels;
        int labelSize;
        cv::Mat edges;
    private:
        void quicksort(std::vector<cv::Point2i> &XYpoints, int p, int r);
        int partition(std::vector<cv::Point2i> &XYpoints, int p, int r);
        void computeCircumcircleOverlap(const std::vector<double>& radii, const std::vector<cv::Point2d>& centers,
                                        const std::vector<double>& neighbors);
        void mxCircleStats(const std::vector<double>& radii, const std::vector<cv::Point2d>& centers,
                           const std::vector<double>& RGBb);
        void computeSquaredEdgeDistances(const double &sigma_c);
        void rgbtolab();
        cv::Mat mat2cvMat(const std::vector<double>& rgb);
        std::vector<double> cvMat2Mat(const cv::Mat &mat);
        bool im2single(const cv::Mat &src, std::vector<double>& mat, const int &plane);
        bool imfilter(const std::vector<double>& src, const std::vector<double>& filter,
                      std::vector<double>& _mat, const bool &isTranspose, const int &plane);
        std::vector<double> dtMat2mat(const std::vector<double>& src);
        std::vector<int> integrate_merges(const std::vector<int>& old_labels,
                                 const std::vector<int>& new_labels,
                                 const double &merge_threshold);
        std::vector<double> compactLabels(const std::vector<int>& tri_labels, const std::vector<double>& labels);
        cv::Mat colorImageSegments(const cv::Mat &bgr, const std::vector<double>& labels, const cv::Scalar &color);
        std::vector<double> accumarray(const std::vector<int>& subs, const std::vector<double>& vals);
        int maxVal(const std::vector<int>& vec);
        int findVal(const std::vector<int>& vec, const int &elem, const int &start);
        void *search(void *arg);
};

#endif
