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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSignalMapper>
#include <QFileDialog>
#include <QGroupBox>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <future>
#include "rgbSegmentation.h"
#include "qtcvtransform.h"
#include "edgedetector.h"
#include "skindetection.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

  public:
      explicit MainWindow(QWidget *parent = 0);

  private:
      std::shared_ptr<Ui::MainWindow> ui;
      QString filename, dirname;
      bool isPossibleToExecute;
      bool isPossibleToExecuteDir;
      bool isSetImages;
      bool automatically;
      bool loadMask;
      cv::Mat rgb, gt, mask, newMask, grid;
      std::vector<std::string> dirlist;
      std::vector<double> error_rate;
      std::vector<double> dpr_rate;
      std::vector<double> fmeasure;
      std::vector<double> accuracies;
      int index;
  private:
      void process();
      int circleCounter(const cv::Mat &img, const int &ncircle);
      int imageColorCounting(const cv::Rect &rect, const cv::Mat &binMask, const cv::Mat &rgb);
      void fprdr(const cv::Mat &resImage, const cv::Mat &gtImage);
      float computeShannonEntropy(const cv::Mat &r, const cv::Mat &g, const cv::Mat &b);
      float getFrequencyOfBin(const cv::Mat &channel);
      float getHistogramBinValue(const cv::Mat &hist, const int &binNum);
      double entropy(const cv::Mat &img);
      void circularity(cv::Mat &src_gray, cv::Point2f c, float r, std::vector<cv::Point2f> vertices,
		      float &tr_ratio, float &br_ratio, float &bl_ratio, float &tl_ratio);

      void stats(cv::Mat maskImage, float &tr_ratio, float &br_ratio, float &bl_ratio, float &tl_ratio);
      void findCircle(cv::Mat &src_gray, std::vector<cv::Point2f> &center, std::vector<float> &radius);
      void findTriangles(cv::Mat &src_gray, std::vector<cv::Point2f> center,
				    std::vector<float> radius, cv::Point2f &c, float &r, std::vector<cv::Point2f> &vertices);

  public slots:
      void open_file();
      void open_directory();
      void open_set();
      void load_mask();
      void compute_segmentation();
      void compute_segmentation_automaticcaly();
      void compute_segmentation_manually();
      void compute_segmentation_automaticcaly_images();
      void next();
      void prev();
};

#endif // MAINWINDOW_H
