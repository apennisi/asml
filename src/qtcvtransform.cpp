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

#include "qtcvtransform.h"

QImage* Mat2QImage(const cv::Mat img)
{
    int h = img.rows;
    int w = img.cols;
    QImage *qimg;

    int channels = img.channels();

    switch(channels) {
        case 1:
            //qimg = new QImage(w, h, QImage::Format_Indexed8);
            //break;
        case 3:
            qimg = new QImage(w, h, QImage::Format_RGB888);
            break;
        default:
            qimg = new QImage(w, h, QImage::Format_ARGB32);
    }

//    QImage *qimg = new QImage(w, h, QImage::Format_ARGB32);

    //cv::imshow("img", img);
    //cv::waitKey(0);

    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            char r, g, b, a = 0;
            if(channels == 1) {
                r = (int)img.at<uchar>(i,j);
                g = (int)img.at<uchar>(i,j);
                b = (int)img.at<uchar>(i,j);
            } else if(channels == 3) {
                r = img.at<cv::Vec3b>(i,j)[2];
                g = img.at<cv::Vec3b>(i,j)[1];
                b = img.at<cv::Vec3b>(i,j)[0];
            }

            qimg->setPixel(j, i, qRgb(r, g, b));
        }
    }

    return qimg;
}


cv::Mat QImage2Mat(const QImage *qimg) {
    cv::Mat mat = cv::Mat(qimg->height(), qimg->width(), CV_8UC4, (uchar*)qimg->bits(), qimg->bytesPerLine());
    cv::Mat mat2 = cv::Mat(mat.rows, mat.cols, CV_8UC3 );
    int from_to[] = { 0,0,  1,1,  2,2 };
    cv::mixChannels( &mat, 1, &mat2, 1, from_to, 3 );
    return mat2;
}
