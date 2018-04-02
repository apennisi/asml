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
#ifndef QTCVTRANSFORM_H
#define QTCVTRANSFORM_H

#include <QApplication>
#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include <string>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <sstream>


cv::Mat QImage2Mat(const QImage *qimg);
QImage*  Mat2QImage(const cv::Mat img);

#endif // QTCVTRANSFORM_H
