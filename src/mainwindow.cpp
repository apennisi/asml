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

#include "mainwindow.h"
#include "ui_mainwindow.h"

int countImages = 0;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(std::shared_ptr<Ui::MainWindow>(new Ui::MainWindow))
{
    ui->setupUi(this);
    ui->edge_detection_label->setStyleSheet("background-color: black");
    ui->skin_detection_label->setStyleSheet("background-color: black");
    ui->blob_detection_label->setStyleSheet("background-color: black");
    ui->rgb_label->setStyleSheet("background-color: black");
    ui->rgb_segmentation_label->setStyleSheet("background-color: black");

    connect(ui->openButton , SIGNAL( clicked() ), this, SLOT( open_file() ) );
    connect(ui->openButtonDir , SIGNAL( clicked() ), this, SLOT( open_directory() ) );
    connect(ui->loadMask , SIGNAL( clicked() ), this, SLOT( load_mask() ) );
    connect(ui->openButtonSet , SIGNAL( clicked() ), this, SLOT( open_set() ) );
    isPossibleToExecute = isPossibleToExecuteDir = automatically = false;

    connect(ui->computeButton, SIGNAL( clicked()), this, SLOT( compute_segmentation() ) );
    connect(ui->computeButtonAuto, SIGNAL( clicked()), this, SLOT( compute_segmentation_automaticcaly() ) );
    connect(ui->computeButtonManually, SIGNAL( clicked()), this, SLOT( compute_segmentation_manually() ) );
    connect(ui->pushButtonNext, SIGNAL( clicked()), this, SLOT( next() ) );
    connect(ui->pushButtonPrev, SIGNAL( clicked()), this, SLOT( prev() ) );

    ui->threshSpinBox->setRange(0.01, 0.09);
    ui->threshSpinBox->setSingleStep(0.01);
    ui->threshSpinBox->setValue(0.03);

    ui->groupBox->setStyleSheet(QString::fromUtf8("QGroupBox { border: 2px solid #000; border-radius: 6px; } QGroupBox::title {background-color: transparent; subcontrol-position: top left; padding:2 13px;}"));
    ui->groupBox_2->setStyleSheet(QString::fromUtf8("QGroupBox { border: 2px solid #000; border-radius: 6px; } QGroupBox::title {background-color: transparent; subcontrol-position: top left; padding:2 13px;}"));

    ui->sigmaSpinBox->setRange(1., 15.);
    ui->sigmaSpinBox->setSingleStep(1.);
    ui->sigmaSpinBox->setValue(2.);

    ui->sigmaBlurringSpinBox->setRange(1., 15.);
    ui->sigmaBlurringSpinBox->setSingleStep(1.);
    ui->sigmaBlurringSpinBox->setValue(5.);

    ui->sigmaCSpinBox->setRange(1., 15.);
    ui->sigmaCSpinBox->setSingleStep(1.);
    ui->sigmaCSpinBox->setValue(3.);

    ui->mergeThresholdSpinBox->setRange(0.1, 1.);
    ui->mergeThresholdSpinBox->setSingleStep(0.05);
    ui->mergeThresholdSpinBox->setValue(0.70);//0.80

    ui->computeButton->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->computeButtonAuto->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->computeButtonManually->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->computeButtonAutomaticImages->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->pushButtonNext->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->pushButtonPrev->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");

    loadMask = false;
    isSetImages = false;
}

void MainWindow::fprdr(const cv::Mat &resImage, const cv::Mat &gtImage) {
    int false_positives = 0;
    int false_negatives = 0;
    int true_positives = 0;
    int true_negatives = 0;

    cv::Mat diff = cv::Mat::zeros(resImage.size(), CV_8UC3);

    for(int i = 0; i < resImage.rows; i++) {
        for(int j = 0; j < resImage.cols; j++) {
            uchar a = (uchar)resImage.at<uchar>(i,j);
            uchar b = (uchar)gtImage.at<uchar>(i,j);

            if(a == 255 && b == 0){
                false_positives++;
                diff.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
            }
            else if(a == 255 && b == 255){
                true_positives++;
                diff.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }
            else if(a == 0 && b == 0 ) {
                true_negatives++;
                diff.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
            }
            else if(a == 0 && b == 255){
                false_negatives++;
                diff.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
            else {
                std::cout <<"error RESULT = "<<(int)a<<" GT = "<<(int)b<< std::endl;
                std::cout <<"at i = "<<i<<" j = "<<j<< std::endl;
                return;
            }

        }
    }

    cv::imshow("Comparison", diff);

    std::stringstream ss;
    ss << "comparison_" << (countImages - 1) << ".png";

   // cv::imwrite(ss.str(), diff);

    double detection_ratio = (double)true_positives / (true_positives+false_negatives);
    double false_alarm_ratio = (double)false_positives / (true_negatives+false_positives);

    std::cout << "Detection: " << detection_ratio <<  " False Alarm: " << false_alarm_ratio << std::endl;

    error_rate.push_back(false_alarm_ratio);
    dpr_rate.push_back(detection_ratio);

    double prec = (true_positives !=0 || false_positives != 0) ?double(true_positives) / double(true_positives + false_positives) : 0;
    double rec = (true_positives !=0 || false_negatives != 0) ? double(true_positives) / double(true_positives + false_negatives) : 0;

    double fmes = (prec != 0 || rec != 0) ? 2 * (prec * rec) / (prec + rec) : 0;

    fmeasure.push_back(fmes);

    double acc = (true_positives != 0 || false_negatives != 0 || true_negatives != 0 || false_positives != 0) ? double(true_negatives + true_positives) / double(true_positives + false_negatives + true_negatives + false_positives) : 0;

    accuracies.push_back(acc);

}

int MainWindow::circleCounter(const cv::Mat &img, const int &ncircle) {

    cv::Mat tempGray, gray;

    cv::Mat sel_1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(200, 200));
    cv::morphologyEx(img.clone(), tempGray, cv::MORPH_ELLIPSE,  sel_1);

    bool whiteContr = false;
    for(int i = 0; i < tempGray.rows; i++) {
        for(int j = 0; j < tempGray.cols; j++) {
            if((uchar)tempGray.at<uchar>(i, j) == 255) {
                whiteContr = true;
            }
        }
    }

    if(!whiteContr) {
        cv::dilate(img.clone(), tempGray, cv::Mat(), cv::Point(-1, -1), 50, 2, 1);
    }

    gray = tempGray.clone();

   // cv::imshow("neo_gray", gray);

    std::vector<cv::Vec3f> circles;

    /// Apply the Hough Transform to find the circles
    cv::HoughCircles( gray, circles, CV_HOUGH_GRADIENT, 1, 90, 255, ncircle, 100, 0 ); //20

    return circles.size();

}

int MainWindow::imageColorCounting(const cv::Rect &rect, const cv::Mat &binMask, const cv::Mat &rgb) {

    cv::Mat temp = cv::Mat::zeros(rgb.size(), rgb.type());
    rgb.copyTo(temp, binMask);

    cv::Mat gg;
    cv::cvtColor(temp, gg, CV_BGR2GRAY);

    cv::Vec3b prevColor(0,0,0);
    int countColor = 0;
    for(int i = rect.tl().y; i <  rect.br().y; ++i)
    {
        for(int j = rect.tl().x; j < rect.br().x; ++j)
        {
            if(temp.at<cv::Vec3b>(i, j) != prevColor && temp.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0))
            {
                prevColor = temp.at<cv::Vec3b>(i, j);
                ++countColor;
            }
        }
    }

    return countColor;
}


void MainWindow::stats(cv::Mat maskImage, float &tr_ratio, float &br_ratio, float &bl_ratio, float &tl_ratio)
{
    cv::Size extendedSize = maskImage.size();

    int offset_w = 1280;
    int offset_h = 960;

    extendedSize.width += offset_w;
    extendedSize.height += offset_h;
    cv::Mat extendedMask = cv::Mat::zeros(extendedSize,CV_8UC1);

    for(int i = offset_h/2; i < (extendedMask.rows - offset_h/2); ++i) {
        for(int j = offset_w/2; j < (extendedMask.cols - offset_w/2); ++j) {
            extendedMask.at<uchar>(i,j) = maskImage.at<uchar>(i-(offset_h/2),j-(offset_w/2));
        }
    }

    std::vector<cv::Point2f> center;
    std::vector<float> radius;
    findCircle(extendedMask, center, radius);

    //cout << "center.size() " << center.size() << endl;

    cv::Point2f c;
    float r;
    std::vector<cv::Point2f> vertices;
    findTriangles(extendedMask, center, radius, c, r, vertices);

    circularity(extendedMask, c, r, vertices, tr_ratio, br_ratio, bl_ratio, tl_ratio);

    cv::Mat resizedMask = cv::Mat::zeros(maskImage.size(),CV_8UC1);
    cv::resize(extendedMask, resizedMask, resizedMask.size());


    cv::namedWindow("extended");
    cv::imshow("extended", resizedMask);
    //cv::waitKey(0);

}

void MainWindow::findCircle(cv::Mat &src_gray, std::vector<cv::Point2f> &center, std::vector<float> &radius) {

    cv::Mat threshold_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    /// Detect edges using Threshold
    cv::threshold( src_gray, threshold_output, 120, 255, CV_THRESH_BINARY );
    /// Find contours
    cv::findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    center.resize( contours.size() );
    radius.resize( contours.size() );

    for( int i = 0; i < int(contours.size()); i++ ) {
        cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
        cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
    }

}

void MainWindow::findTriangles(cv::Mat &src_gray, std::vector<cv::Point2f> center,
                               std::vector<float> radius, cv::Point2f &c, float &r, std::vector<cv::Point2f> &vertices) {

    r = -1;
    for( int i = 0; i< center.size(); i++ ) {
        if(radius[i] > r) {
            r = radius[i];
            c = center [i];
        }
    }

    if(r > 0.)
        cv::circle( src_gray, c, cvRound(r), cv::Scalar(120), 2, 8, 0 );

    //top right
    int tr_y = -1;
    for(int i = c.y; i > 0; --i) {
        if(src_gray.at<uchar>(i, c.x) == 120) {
            tr_y = i;
            break;
        }
    }
    if(tr_y < 0) {
        std::cerr << "Unable to find tr_y" << std::endl;
        std::cerr <<"exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    int tr_x = -1;
    for(int j = c.x; j < src_gray.cols; ++j) {
        if(src_gray.at<uchar>(c.y, j) == 120) {
            tr_x = j;
            break;
        }
    }
    if(tr_x < 0) {
        std::cerr << "Unable to find tr_x" << std::endl;
        std::cerr <<"exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    vertices.push_back(cv::Point2f(tr_x, tr_y));


    //bottom right
    int br_y = -1;
    for(int i = c.y; i < src_gray.rows; ++i) {
        if(src_gray.at<uchar>(i, c.x) == 120) {
            br_y = i;
            break;
        }
    }
    if(br_y < 0) {
        std::cerr << "Unable to find br_y" << std::endl;
        std::cerr <<"exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    int br_x = tr_x;
    vertices.push_back(cv::Point2f(br_x, br_y));

    //bottom left
    int bl_y = br_y;
    int bl_x = -1;
    for(int j = c.x; j > 0; --j) {
        if(src_gray.at<uchar>(c.y, j) == 120) {
            bl_x = j;
            break;
        }
    }
    if(bl_x < 0) {
        std::cerr << "Unable to find bl_x" << std::endl;
        std::cerr <<"exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    vertices.push_back(cv::Point2f(bl_x, bl_y));

    //top left
    int tl_y = tr_y;
    int tl_x = bl_x;
    vertices.push_back(cv::Point2f(tl_x, tl_y));


    //center
    //circle( src_gray, c, 6, Scalar(100), -1, 8, 0 );

    //top right hypotenuse
    //circle( src_gray, Point(c.x, tr_y), 6, Scalar(100), -1, 8, 0 );
    //circle( src_gray, Point(tr_x, c.y), 6, Scalar(100), -1, 8, 0 );
    cv::line(src_gray, cv::Point(c.x, tr_y), cv::Point(tr_x, c.y), cv::Scalar(80), 3);

    //bottom right hypotenuse
    //circle( src_gray, Point(c.x, br_y), 6, Scalar(100), -1, 8, 0 );
    cv::line(src_gray, cv::Point(c.x, br_y), cv::Point(br_x, c.y), cv::Scalar(80), 3);

    //bottom left hypotenuse
    //circle( src_gray, Point(bl_x, c.y), 6, Scalar(100), -1, 8, 0 );
    cv::line(src_gray, cv::Point(c.x, bl_y), cv::Point(bl_x, c.y), cv::Scalar(80), 3);

    //top left hypotenuse
    cv::line(src_gray, cv::Point(c.x, tl_y), cv::Point(tl_x, c.y), cv::Scalar(80), 3);

}

void MainWindow::circularity(cv::Mat &src_gray, cv::Point2f c, float r, std::vector<cv::Point2f> vertices,
                 float &tr_ratio, float &br_ratio, float &bl_ratio, float &tl_ratio) {
    //top right
    float b = vertices[0].x - c.x;
    float h = c.y - vertices[0].y;
    float tr_area = (b * h) / 2.0f;

    //cout << "tr_area = " << tr_area << endl;

    int tr_cnt = 0;
    for(int i = vertices[0].y; i < c.y; ++i) {
        for(int j = c.x; j < vertices[0].x; ++j) {
            if(src_gray.at<uchar>(i,j) == 255) {
                ++tr_cnt;


                src_gray.at<uchar>(i,j) = 30;

            }
            else if(src_gray.at<uchar>(i,j) == 80) {

                src_gray.at<uchar>(i,j) = 240;

                break;
            }
        }
    }

    tr_ratio = tr_cnt/tr_area;

    //cout << "tr_cnt = " << tr_cnt << endl;
    //cout << "tr_ratio = " << tr_ratio << endl;


    //bottom right
    b = vertices[1].x - c.x;
    h = vertices[1].y - c.y;
    float br_area = (b * h) / 2.0f;

    //cout << "br_area = " << br_area << endl;

    int br_cnt = 0;
    for(int i = c.y; i < vertices[1].y; ++i) {
        for(int j = c.x; j < vertices[1].x; ++j) {
            if(src_gray.at<uchar>(i,j) == 255) {
                ++br_cnt;

                src_gray.at<uchar>(i,j) = 30;

            }
            else if(src_gray.at<uchar>(i,j) == 80) {

                src_gray.at<uchar>(i,j) = 240;


                break;
            }
        }
    }

    br_ratio = br_cnt/br_area;

    //bottom left
    b = c.x - vertices[2].x;
    h = vertices[2].y - c.y;
    float bl_area = (b * h) / 2.0f;

    int bl_cnt = 0;
    for(int i = c.y; i < vertices[2].y; ++i)
    {
        for(int j = c.x; j > vertices[2].x; --j)
        {
            if(src_gray.at<uchar>(i,j) == 255)
            {
                ++bl_cnt;
                src_gray.at<uchar>(i,j) = 30;

            }
            else if(src_gray.at<uchar>(i,j) == 80)
            {
                src_gray.at<uchar>(i,j) = 240;
                break;
            }
        }
    }

    bl_ratio = bl_cnt/bl_area;

    //cout << "bl_cnt = " << bl_cnt << endl;
    //cout << "bl_ratio = " << bl_ratio << endl;




    //top left
    b = c.x - vertices[3].x;
    h = c.y - vertices[3].y;
    float tl_area = (b * h) / 2.0f;

    //cout << "tl_area = " << tl_area << endl;

    int tl_cnt = 0;
    for(int i = vertices[3].y; i < c.y; ++i) {
        for(int j = c.x; j > vertices[3].x; --j) {
            if(src_gray.at<uchar>(i,j) == 255) {
                ++tl_cnt;

                src_gray.at<uchar>(i,j) = 30;


            }
            else if(src_gray.at<uchar>(i,j) == 80) {

                src_gray.at<uchar>(i,j) = 240;

                break;
            }
        }
    }

    tl_ratio = tl_cnt/tl_area;
}

void MainWindow::process() {

    //Compute Skin Detection
    SkinDetection skindetector(rgb.clone());
    skindetector.compute();
    cv::Mat neo = skindetector.getImg();

    cv::Mat binNeo = skindetector.getBw();

    cv::Mat gray;
    //cv::cvtColor(neo, gray, CV_BGR2GRAY);

    gray = binNeo.clone();

    int nCircles = circleCounter(gray, 10);

    cv::Mat dst_neo = cv::Mat::zeros(cv::Size(ui->skin_detection_label->width(),
                                          ui->skin_detection_label->height()), neo.type());
    cv::resize(neo, dst_neo, dst_neo.size());

    QImage *tmpImg_neo = Mat2QImage(dst_neo);

    QPixmap image_neo = QPixmap::fromImage(*tmpImg_neo);

    ui->skin_detection_label->setPixmap(image_neo);

    delete tmpImg_neo;

    //EQ
    std::vector<cv::Mat> channels;
    cv::Mat img_hist_equalized;

    cv::cvtColor(rgb, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format

    cv::split(img_hist_equalized,channels); //split the image into channels

    cv::equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

    cv::merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

    cv::cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR);

    cv::Mat rgb_new;
    rgb_new = img_hist_equalized.clone();

    //cv::imshow("eq", img_hist_equalized);

    //Hair Removal
    cv::Mat sel = cv::getStructuringElement(cv::MORPH_OPEN , cv::Size(9,9)); //15, 15
    cv::morphologyEx(rgb_new, rgb_new, cv::MORPH_DILATE,  sel);

    cv::GaussianBlur(rgb_new, rgb_new, cv::Size(9, 9), 2, 0, cv::BORDER_REPLICATE); //9, 9

    int vv1 = 40;//80

    rgb_new += cv::Scalar(vv1, vv1, vv1);

    //Edge Detection
    cv::Mat grayImg;
    cv::cvtColor(rgb_new.clone(), grayImg, CV_BGR2GRAY);
    cv::Mat edge;

    double thresh = ui->threshSpinBox->value();
    double sigma = ui->sigmaSpinBox->value();

    EdgeDetector edges(grayImg, edge, thresh, sigma);


    //end Edge Detection

    cv::Mat dst = cv::Mat::zeros(cv::Size(ui->edge_detection_label->width(),
                                          ui->edge_detection_label->height()), edge.type());
    cv::resize(edge, dst, dst.size());



    QImage *tmpImg = Mat2QImage(dst);
    QPixmap image = QPixmap::fromImage(*tmpImg);

    ui->edge_detection_label->setPixmap(image);

    delete tmpImg;


    //RGB Segmentation
    std::vector<cv::Point2i> XYpoints = edges.getEdgesPoints();


    double sigma_blurring = ui->sigmaBlurringSpinBox->value();
    double sigma_c = ui->sigmaCSpinBox->value();
    double merge_threshold = ui->mergeThresholdSpinBox->value();

    cv::Mat rgbLabels;
    RgbSegmentation rgb_seg(rgb_new, rgb, XYpoints, sigma_blurring, sigma_c, merge_threshold, rgbLabels);

    cv::Mat bin_edges = cv::Mat::zeros(rgbLabels.size(), CV_8UC1);

    for(int i = 0; i < rgbLabels.rows; ++i)
    {
        for(int j = 0; j < rgbLabels.cols; ++j)
        {
            if(rgbLabels.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 255))
            {
                bin_edges.at<uchar>(i, j) = 255;
                rgbLabels.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }




    std::vector < std::vector<cv::Point> > contours, filteredContours;

    cv::findContours(bin_edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    bin_edges = cv::Scalar(0);


    std::vector<std::vector<cv::Point> > contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());

    cv::Mat tempCont = bin_edges.clone();

    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
    {
        tempCont = cv::Scalar(0);
        cv::approxPolyDP(cv::Mat(contours[contourIdx]), contours_poly[contourIdx], 3, true);
        boundRect[contourIdx] = cv::boundingRect(cv::Mat(contours_poly[contourIdx]));
        cv::drawContours( tempCont, contours, contourIdx, cv::Scalar(255), CV_FILLED );

        int pixelCount = 0;
        int thresh_color = 190;

        for(size_t i = boundRect[contourIdx].tl().y; i < boundRect[contourIdx].br().y; ++i)
        {
            for(size_t j = boundRect[contourIdx].tl().x; j < boundRect[contourIdx].br().x; ++j)
            {

                if(tempCont.at<uchar>(i, j) == uchar(255) && ((rgbLabels.at<cv::Vec3b>(i, j)[0] >= thresh_color &&
                     rgbLabels.at<cv::Vec3b>(i, j)[1] >= thresh_color) ||  (rgbLabels.at<cv::Vec3b>(i, j)[0] >= thresh_color &&
                     rgbLabels.at<cv::Vec3b>(i, j)[2] >= thresh_color) ||  (rgbLabels.at<cv::Vec3b>(i, j)[1] >= thresh_color &&
                                                                  rgbLabels.at<cv::Vec3b>(i, j)[2] >= thresh_color)))
                {
                    ++pixelCount;
                    rgbLabels.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
        }

    }

    cv::Mat tempRgbLabels = cv::Mat::zeros(rgbLabels.size(), CV_8UC3),
            tempRgbLabelsBw, new_edge;

    rgbLabels.copyTo(tempRgbLabels, newMask);

    cv::cvtColor(tempRgbLabels, tempRgbLabelsBw, CV_RGB2GRAY);

    EdgeDetector new_edges(tempRgbLabelsBw, new_edge, thresh, sigma);

    cv::dilate(new_edge, new_edge, cv::Mat());

    cv::findContours(new_edge, filteredContours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);



    double maxArea = ((rgbLabels.rows*rgbLabels.cols)/2) - 1000, minArea = 5000;
    cv::Moments moms = cv::moments(cv::Mat(filteredContours[0]));
    double bestArea = (moms.m00 < maxArea && moms.m00 > minArea) ? moms.m00 :  minArea;
    int idx = 0;


    for (size_t contourIdx = 1; contourIdx < filteredContours.size(); ++contourIdx)
    {
        std::vector<cv::Point> poly;
        cv::approxPolyDP(cv::Mat(filteredContours[contourIdx]), poly, 3, true);
        cv::Rect bbox = cv::boundingRect(cv::Mat(poly));
        double ratio = (bbox.width > bbox.height) ? double(bbox.width) / double(bbox.height) : double(bbox.height) / double(bbox.width);
        moms = cv::moments(cv::Mat(filteredContours[contourIdx]));
        double area = moms.m00;

        bin_edges = cv::Scalar(0);
        cv::drawContours( bin_edges, filteredContours, contourIdx, cv::Scalar(255), CV_FILLED );

        if(area > bestArea && area < maxArea && area > minArea && std::abs(bbox.width - bbox.height) < 350)
        {
            bin_edges = cv::Scalar(0);
            cv::drawContours( bin_edges, filteredContours, contourIdx, cv::Scalar(255), CV_FILLED );

            int nTempCircles = circleCounter(bin_edges, 8);
            int colorCounter = imageColorCounting(bbox, bin_edges, rgbLabels);

            if(nTempCircles > 0 && colorCounter > 1500)
            {
                bestArea = area;
                idx = contourIdx;
            }
        }
    }

    bin_edges = cv::Scalar(0);

    cv::drawContours( bin_edges, filteredContours, idx, cv::Scalar(255), CV_FILLED );

    int segCircleNumber = circleCounter(bin_edges, 8);

    cv::Mat dst_2 = cv::Mat::zeros(cv::Size(ui->edge_detection_label->width(),
                                                  ui->edge_detection_label->height()), rgbLabels.type());
    cv::resize(rgbLabels, dst_2, dst.size());

    QImage *tmpImg_2 = Mat2QImage(dst_2);

    QPixmap image_rgb = QPixmap::fromImage(*tmpImg_2);

    ui->rgb_segmentation_label->setPixmap(image_rgb);

    delete tmpImg_2;

    cv::Mat tempFusion;
    if(nCircles != 0 && segCircleNumber != 0)
        cv::bitwise_and(bin_edges, binNeo, tempFusion);
    else if(segCircleNumber != 0)
        tempFusion = bin_edges.clone();
    else {
        tempFusion = binNeo.clone();
    }

    cv::Mat dst_fusion = cv::Mat::zeros(cv::Size(ui->edge_detection_label->width(),
                                          ui->edge_detection_label->height()), edge.type());
    cv::resize(tempFusion, dst_fusion, dst.size());



    QImage *tmpImgFusion = Mat2QImage(dst_fusion);
    QPixmap image_fusion = QPixmap::fromImage(*tmpImgFusion);

    ui->blob_detection_label->setPixmap(image_fusion);


    delete tmpImgFusion;

}


void MainWindow::compute_segmentation() {
    if(isPossibleToExecute) {
        process();
    }
}

void MainWindow::next() {
    if(isPossibleToExecuteDir) {
        std::cout << index << " " << dirlist.size() << std::endl;
        (index < dirlist.size()) ? index++ : index = index;
        compute_segmentation_manually();
    }
}

void MainWindow::prev() {
    if(isPossibleToExecuteDir) {
        (index == 0) ? index = 0 : index--;
        compute_segmentation_manually();
    }
}

void MainWindow::compute_segmentation_manually() {
    if(isPossibleToExecuteDir) {
        ui->pushButtonNext->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");
        ui->pushButtonPrev->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");
        std::string dir = dirname.toUtf8().constData();
        std::string im;
        if(isSetImages) {
            im = dir + "/" + dirlist[index];
        } else
            im = dir + "/" + dirlist[index] + "/" + dirlist[index] +
                "_Dermoscopic_Image/" + dirlist[index] + ".bmp";
#ifdef GT
        std::string gt_name = dir + "/" + dirlist[index] + "/" + dirlist[index] +
                "_lesion/" + dirlist[index] + "_lesion.bmp";
#endif
        //cv::Mat rgbTmp = cv::imread(im.c_str());

        //rgb = cv::Scalar(0, 0, 0);
        rgb = cv::imread(im.c_str());

        if(loadMask) {
            newMask = cv::Mat::zeros(rgb.size(), CV_8UC1);

            cv::resize(mask, newMask, newMask.size());

            cv::Mat temp = rgb.clone();

            rgb = cv::Scalar(0, 0, 0);

            temp.copyTo(rgb, newMask);

        }

        std::cout << dir << std::endl;
#ifdef GT
        gt = cv::imread(gt_name.c_str(), CV_LOAD_IMAGE_UNCHANGED);
#endif

        cv::Mat dst = cv::Mat::zeros(cv::Size(ui->rgb_label->width(),
                                              ui->rgb_label->height()), rgb.type());
        cv::resize(rgb, dst, dst.size());

        QImage *tmpImg = Mat2QImage(dst);
        QPixmap image = QPixmap::fromImage(*tmpImg);

        ui->rgb_label->setPixmap(image);

        delete tmpImg;

        process();
    }
}

void MainWindow::compute_segmentation_automaticcaly_images()
{

}

void MainWindow::compute_segmentation_automaticcaly() {
    if(isPossibleToExecuteDir) {
        int end = dirlist.size();
        std::string dir = dirname.toUtf8().constData();
        automatically = true;
        for(int i = 0; i < end; ++i) {
            std::string im = dir + "/" + dirlist[i] + "/" + dirlist[i] +
                    "_Dermoscopic_Image/" + dirlist[i] + ".bmp";
            std::string gt_name = dir + "/" + dirlist[i] + "/" + dirlist[i] +
                    "_lesion/" + dirlist[i] + "_lesion.bmp";


            rgb = cv::imread(im.c_str());

            if(loadMask) {
                newMask = cv::Mat::zeros(rgb.size(), CV_8UC1);

                cv::resize(mask, newMask, newMask.size());

                rgb.copyTo(rgb, newMask);
            }

            gt = cv::imread(gt_name.c_str(), CV_LOAD_IMAGE_UNCHANGED);

            cv::Mat dst = cv::Mat::zeros(cv::Size(ui->rgb_label->width(),
                                                  ui->rgb_label->height()), rgb.type());
            cv::resize(rgb, dst, dst.size());

            QImage *tmpImg = Mat2QImage(dst);
            QPixmap image = QPixmap::fromImage(*tmpImg);

            ui->rgb_label->setPixmap(image);

            delete tmpImg;

            process();

        }

        double dr = 0., fr = 0.;
        for(int i = 0; i < error_rate.size(); ++i) {
            dr += dpr_rate[i];
            fr += error_rate[i];
        }

        std::cout << "TOT DR: " << dr / error_rate.size() << " TOT FR: " << fr / error_rate.size() << std::endl;

        double acc = 0.;
        for(int i = 0; i < accuracies.size(); ++i) {
            acc += accuracies[i];
        }

        std::cout << "Accuracy: " << acc/accuracies.size() << std::endl;

        double accuracy = acc/accuracies.size();

        double variance = 0.;
        for(int i = 0; i < accuracies.size(); ++i) {
            variance += ( (accuracies[i] - accuracy)*(accuracies[i] - accuracy) );
        }

        std::cout << "Variance: " << variance/accuracies.size() << std::endl;

        double fmeas = 0.;
        for(int i = 0; i < fmeasure.size(); ++i) {
            fmeas += fmeasure[i];
        }

        fmeas /= fmeasure.size();

        std::cout << "F-measure: " << fmeas << std::endl;

    }
}

double MainWindow::entropy(const cv::Mat &img)
{
    int numBins = 256, nPixels;
    float range[] = {0, 255};
    double imgEntropy = 0, prob;
    const float* histRange = { range };
    cv::Mat histValues;

    cv::calcHist(&img, 1, 0, cv::Mat(), histValues, 1, &numBins, &histRange, true, true );

    nPixels = sum(histValues)[0];


    for(int i = 1; i < numBins; i++)
    {
        prob = histValues.at<float>(i)/nPixels;
        if(prob < FLT_EPSILON)
            continue;
        imgEntropy += prob*(log(prob)/log(2));

    }

    return -imgEntropy;
}

float MainWindow::getHistogramBinValue(const cv::Mat &hist, const int &binNum) {
    return float(hist.at<uchar>(binNum));
}

float MainWindow::getFrequencyOfBin(const cv::Mat &channel) {
    float frequency = 0.0;
    for( int i = 1; i < 255; i++ ) {
        float Hc = abs(getHistogramBinValue(channel,i));
        frequency += Hc;
    }
    std::cout << "frequency: " << frequency << std::endl;
    return frequency;
}

float MainWindow::computeShannonEntropy(const cv::Mat &r, const cv::Mat &g, const cv::Mat &b)
{
    float entropy = 0.0;
    float frequency = getFrequencyOfBin(r);
    for( int i = 1; i < 255; i++ )
    {
        float Hc = abs(getHistogramBinValue(r,i));
        entropy += -(Hc/frequency) * log10((Hc/frequency));
    }
    std::cout << "Entropy 1: " << entropy << std::endl;
    frequency = getFrequencyOfBin(g);
    for( int i = 1; i < 255; i++ )
    {
        float Hc = abs(getHistogramBinValue(g,i));
        entropy += -(Hc/frequency) * log10((Hc/frequency));
    }
    std::cout << "Entropy 2: " << entropy << std::endl;
    frequency = getFrequencyOfBin(b);
    for( int i = 1; i < 255; i++ )
    {
        float Hc = abs(getHistogramBinValue(b,i));
        entropy += -(Hc/frequency) * log10((Hc/frequency));
    }

    //cout << entropy <<endl;
    std::cout << "Entropy 3: " << entropy << std::endl;
    return entropy;
}

void MainWindow::open_set() {

    if(isPossibleToExecute) {
        isPossibleToExecute = false;
        ui->computeButton->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
        ui->edge_detection_label->clear();
        ui->edge_detection_label->setStyleSheet("background-color: black");
        ui->skin_detection_label->clear();
        ui->skin_detection_label->setStyleSheet("background-color: black");
        ui->blob_detection_label->clear();
        ui->blob_detection_label->setStyleSheet("background-color: black");
        ui->rgb_label->clear();
        ui->rgb_label->setStyleSheet("background-color: black");
        ui->rgb_segmentation_label->clear();
        ui->rgb_segmentation_label->setStyleSheet("background-color: black");
    }

    ui->computeButtonAuto->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->computeButtonManually->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->computeButtonAutomaticImages->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->pushButtonNext->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->pushButtonPrev->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");

    std::string dirName = ".";

    dirname = QFileDialog::getExistingDirectory(this,
                                                tr("Choose a Directory"),
                                                dirName.c_str(),
                                                QFileDialog::ShowDirsOnly);

    index = 0;

    if(std::string(dirname.toUtf8().constData()) != "") {
        isPossibleToExecuteDir = true;
        ui->computeButtonAuto->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");
        ui->computeButtonManually->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");
        ui->computeButtonAutomaticImages->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");

        dirlist.clear();

        DIR *dir = opendir(dirname.toUtf8().constData());
        struct dirent *dp;
        while ((dp=readdir(dir)) != NULL) {
            if(strcmp(dp->d_name, "..") != 0  &&  strcmp(dp->d_name, ".") != 0 && dp->d_name[0] != '.'
                    && dp->d_name[0] != '~') {
                dirlist.push_back(std::string(dp->d_name));
            }
        }

        std::sort(dirlist.begin(), dirlist.end());

        isSetImages = true;

    } else {
        isPossibleToExecuteDir = false;
        automatically = false;
    }

}

void MainWindow::open_directory() {

    if(isPossibleToExecute) {
        isPossibleToExecute = false;
        ui->computeButton->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
        ui->edge_detection_label->clear();
        ui->edge_detection_label->setStyleSheet("background-color: black");
        ui->skin_detection_label->clear();
        ui->skin_detection_label->setStyleSheet("background-color: black");
        ui->blob_detection_label->clear();
        ui->blob_detection_label->setStyleSheet("background-color: black");
        ui->rgb_label->clear();
        ui->rgb_label->setStyleSheet("background-color: black");
        ui->rgb_segmentation_label->clear();
        ui->rgb_segmentation_label->setStyleSheet("background-color: black");
    }

    ui->computeButtonAuto->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->computeButtonManually->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->pushButtonNext->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
    ui->pushButtonPrev->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");

    std::string dirName = ".";

    dirname = QFileDialog::getExistingDirectory(this,
                                                tr("Choose a Directory"),
                                                dirName.c_str(),
                                                QFileDialog::ShowDirsOnly);

    index = 0;

    if(std::string(dirname.toUtf8().constData()) != "") {
        isPossibleToExecuteDir = true;
        ui->computeButtonAuto->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");
        ui->computeButtonManually->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");

        dirlist.clear();

        DIR *dir = opendir(dirname.toUtf8().constData());
        struct dirent *dp;
        while ((dp=readdir(dir)) != NULL) {
            if(strcmp(dp->d_name, "..") != 0  &&  strcmp(dp->d_name, ".") != 0 && dp->d_name[0] != '.'
                    && dp->d_name[0] != '~') {
                dirlist.push_back(std::string(dp->d_name));
            }
        }

        std::sort(dirlist.begin(), dirlist.end());

    } else {
        isPossibleToExecuteDir = false;
        automatically = false;
    }

}

void MainWindow::load_mask() {

    std::string dirName = ".";

    filename = QFileDialog::getOpenFileName(
                this,
                tr("Open Document"),
                dirName.c_str(),
                tr("Images (*.jpg *.png *.bmp);;") );

    std::string file = filename.toUtf8().constData();

    std::cout << file << std::endl;

    ui->computeButton->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");

    if(file != "") {
        int index = file.find_last_of("/");

        dirName = file.substr(0, index);
        isPossibleToExecute = true;

        mask = cv::imread(file.c_str(), CV_LOAD_IMAGE_ANYDEPTH);

        cv::imshow("mask", mask);

        loadMask = true;
    }

}

void MainWindow::open_file() {

    if(isPossibleToExecuteDir) {
        isPossibleToExecuteDir = false;
        ui->computeButtonAuto->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
        ui->computeButtonManually->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
        ui->pushButtonNext->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
        ui->pushButtonPrev->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");
        ui->edge_detection_label->clear();
        ui->edge_detection_label->setStyleSheet("background-color: black");
        ui->skin_detection_label->clear();
        ui->skin_detection_label->setStyleSheet("background-color: black");
        ui->blob_detection_label->clear();
        ui->blob_detection_label->setStyleSheet("background-color: black");
        ui->rgb_label->clear();
        ui->rgb_label->setStyleSheet("background-color: black");
        ui->rgb_segmentation_label->clear();
        ui->rgb_segmentation_label->setStyleSheet("background-color: black");
    }

    std::string dirName = ".";

    filename = QFileDialog::getOpenFileName(
                this,
                tr("Open Document"),
                dirName.c_str(),
                tr("Images (*.jpg *.png *.bmp);;") );

    std::string file = filename.toUtf8().constData();

    std::cout << file << std::endl;

    ui->computeButton->setStyleSheet("color: rgb(120, 120, 120); color: rgb(120, 120, 120)");

    if(file != "") {
        int index = file.find_last_of("/");

        dirName = file.substr(0, index);
        isPossibleToExecute = true;

        rgb = cv::imread(file.c_str());
        cv::Mat dst = cv::Mat::zeros(cv::Size(ui->rgb_label->width(),
                                              ui->rgb_label->height()), rgb.type());
        cv::resize(rgb, dst, dst.size());

        QImage *tmpImg = Mat2QImage(dst);
        QPixmap image = QPixmap::fromImage(*tmpImg);

        ui->rgb_label->setPixmap(image);

        delete tmpImg;

        ui->computeButton->setStyleSheet("color: rgb(0, 0, 0); color: rgb(0, 0, 0)");
    } else {
        isPossibleToExecute = false;
    }
}
