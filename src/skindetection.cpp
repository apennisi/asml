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

#include "skindetection.h"

SkinDetection::SkinDetection(const cv::Mat &img)
{
    this->img = img;

    Y_MIN  = 0;
    Y_MAX  = 255;
    Cr_MIN = 133;
    Cr_MAX = 145;
    Cb_MIN = 77;
    Cb_MAX = 127;

    Hmin = 0;
    Hmax = 160;
    smin = 100;
    smax = 255;
    vmin = 0; //60
    vmax = 15;
}


void SkinDetection::setParameters(const int &Y, const int &Cr, const int &Cb, const int &H, const int &s, const int &v) {
    Y_MIN = Y;
    Cr_MIN = Cr;
    Cb_MIN = Cb;

    Hmax = H;
    smin = s;
    vmax = v;
}

void SkinDetection::compute() {

    cv::Mat img_skin = img.clone();

    std::vector<cv::Mat> channels;
    cv::Mat img_hist_equalized;

    cv::cvtColor(img, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format

    cv::split(img_hist_equalized,channels); //split the image into channels

    cv::equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

    cv::merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

    cv::cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR);

    img = img_hist_equalized.clone();

    //Hair Removal
    cv::Mat sel = cv::getStructuringElement(cv::MORPH_RECT , cv::Size(11,11));
    cv::morphologyEx(img, img, cv::MORPH_CLOSE,  sel);

    cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::inRange(ycrcb,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),bw);

    img_skin = cv::Scalar(0, 0, 0);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            if((int)bw.at<uchar>(i, j) == 0)
                img_skin.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j);
        }
    }

    cv::cvtColor(img_skin, hsv, CV_BGR2HSV);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j< img.cols; j++) {
            int valueH = hsv.at<cv::Vec3b>(i,j)[2];
            int valueS = hsv.at<cv::Vec3b>(i,j)[1];
            int valueV = hsv.at<cv::Vec3b>(i,j)[0];

            float normH = static_cast<float>(valueH) / static_cast<float>(valueH + valueS + valueV);
            float normS = static_cast<float>(valueS) / static_cast<float>(valueH + valueS + valueV);
            float normV = static_cast<float>(valueV) / static_cast<float>(valueH + valueS + valueV);

            hsv.at<cv::Vec3b>(i, j) = cv::Vec3f( normH*255, normS*255, normV*255 );
        }
    }

    cv::convertScaleAbs(hsv, hsv);

    bw = cv::Scalar(0);

    inRange(hsv, cv::Scalar(Hmin, smin, vmin), cv::Scalar(Hmax, smax, vmax), bw);
    cv::dilate(bw, bw, cv::Mat(), cv::Point(-1, -1), 6);

    cv::erode(bw, bw, cv::Mat(), cv::Point(-1, -1), 2);

    cv::Mat canny_output;

    cv::Canny( bw, canny_output, 50, 200, 3 );
    std::vector < std::vector<cv::Point> > contours;

    cv::findContours(bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    bw = cv::Scalar(0);
    cv::Moments moms = cv::moments(cv::Mat(contours[0]));
    double maxArea = moms.m00;
    int idx = 0;

    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
    {
        moms = cv::moments(cv::Mat(contours[contourIdx]));
        if(moms.m00 > maxArea) {
            maxArea = moms.m00;
            idx = contourIdx;
        }
    }

    cv::drawContours( bw, contours, idx, cv::Scalar(255), CV_FILLED );

    img.copyTo(neo, bw);

    /*int countBw = 0;
    for(int i = 0; i < bw.rows; i++) {
        for(int j = 0; j < bw.cols; j++) {
            if((int)bw.at<uchar>(i, j) == 255) {
                countBw++;
            }
        }
    }

    perc = double(100 * countBw) / double(bw.rows * bw.cols);
    std::cout << "Perc: "<< perc << std::endl;*/
}
