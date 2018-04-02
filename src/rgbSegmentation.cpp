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

#include <rgbSegmentation.h>

RgbSegmentation::RgbSegmentation(const cv::Mat &img, const cv::Mat &rgbImg,
                                 std::vector<cv::Point2i> XYPoints,
                                 const double &sigma_blurring, const double &sigma_c,
                                 const double &merge_threshold, cv::Mat &RgbLabels) {
	
    rows = img.rows;
    cols = img.cols;
    pixelNumber = rows * cols;

    int i, newSize = XYPoints.size() + 4, l, old_size = XYPoints.size();


    std::sort(XYPoints.begin(), XYPoints.end(), RgbSegmentation::compare_xy());

    std::vector<cv::Point2i> newXYPoints(newSize);
    newXYPoints[0] = cv::Point2i(1, 1);
    newXYPoints[1] = cv::Point2i(cols, 1);
    newXYPoints[2] = cv::Point2i(1, rows);
    newXYPoints[3] = cv::Point2i(cols, rows);

    for(i = 4, l = 0; i < newSize && l < old_size; ++i, ++l)
    {
        newXYPoints[i] = cv::Point2i(XYPoints[l].x + 1, XYPoints[l].y + 1);
    }


    //TRIANGULATION
    DelaunayTri d(newXYPoints, cols, rows, newSize);
    sizeTriangles = d.getSize();
    std::vector<double> radii = d.getRadii();
    std::vector<int> indices = d.getIndices();
    neighbors = d.getNeighbors();
    std::vector<cv::Point2d> centers = d.getCenters();


    //CIRCLE OVERLAPPING
    computeCircumcircleOverlap( radii, centers, neighbors );

    const int& width = 2*std::ceil(sigma_blurring) + 1;

    const int& total_width = width*2 + 1;

    gaussKernel = std::vector<double>(total_width);

    double sum = .0;
    int x;
    for(x = -width, i = 0; x < width, i < total_width; ++x, ++i)
    {
        gaussKernel[i]  = std::exp(-(square<double>(x/sigma_blurring)));
        sum += gaussKernel[i];
    }

    for(i = 0; i < total_width; ++i)
    {
        gaussKernel[i] /= sum;
    }


    cv::Mat hsv;

    //aggiungere RGB non img
    cv::cvtColor(rgbImg, hsv, CV_RGB2HSV);

    std::vector<cv::Mat> bgr_planes;
    cv::split( hsv, bgr_planes );

    std::vector<double> bgr_dtMats(pixelNumber*3, 0.);
    std::vector<double> bgr_gaussDtMats(pixelNumber*3, 0.);

    size_t j;
    for(j = 0; j < 3; j++)
    {
        if(!im2single(bgr_planes[j], bgr_dtMats, j))
        {
            std::cerr << "No B&W Image[" << j << "]" << std::endl;
            exit(-1);
        }

        imfilter( bgr_dtMats, gaussKernel, bgr_gaussDtMats, true, j );
        imfilter( bgr_gaussDtMats, gaussKernel, bgr_gaussDtMats, false, j );
    }


    const std::vector<double>& RGBb = dtMat2mat(bgr_gaussDtMats);

    mxCircleStats(radii, centers, RGBb);

    computeSquaredEdgeDistances(sigma_c);

    MergeTriangles mergeTriangles(neighbors, color_weights, edge_lengths, sizeTriangles);
    mergeTriangles.getNewLabels(merge_costs, old_labels, new_labels, merge_lengths);

    const std::vector<int>& tri_labels = integrate_merges(old_labels, new_labels,
                                                merge_threshold);


    //LABELING
    RasterizeTriangles rt(rows, cols, newXYPoints, indices, sizeTriangles);
    const std::vector<double>& labels = rt.getRasterizedImage();

    const std::vector<double>& labels2 = this->compactLabels(tri_labels, labels);

    RgbLabels = colorImageSegments(img, labels2, cv::Scalar(0, 0, 255));
}

RgbSegmentation::~RgbSegmentation()
{

}

cv::Mat RgbSegmentation::colorImageSegments(const cv::Mat &bgr, const std::vector<double>& labels,
                                            const cv::Scalar &color) {

    int i, j, k;
    std::vector<cv::Mat> bgr_planes;
    cv::split(bgr.clone(), bgr_planes);

    std::vector<double> b(rows*cols), g(rows*cols), r(rows*cols);
    std::vector<int> labelVec(rows*cols);
    k = 0;
    uint offset;
    for(j = 0; j < cols; ++j)
    {
        offset = 0;
        for(i = 0; i < rows; ++i)
        {
            b[k] = (double)bgr_planes[0].at<uchar>(i,j) / 255.0000;
            g[k] = (double)bgr_planes[1].at<uchar>(i,j) / 255.0000;
            r[k] = (double)bgr_planes[2].at<uchar>(i,j) / 255.0000;
            labelVec[k] = labels[offset + j];
            ++k;
            offset += cols;
        }
    }

    const std::vector<double>& red_sums = accumarray(labelVec, r);
    const std::vector<double>& green_sums = accumarray(labelVec, g);
    const std::vector<double>& blue_sums = accumarray(labelVec, b);


    int size = labelSize + 1;

    std::vector<double> lut(size * 3);


    lut[0] = (double)color[0] / 255.;
    lut[1] = (double)color[1] / 255.;
    lut[2] = (double)color[2] / 255.;


     for(i = 0; i < labelSize; ++i)
     {
        j = i + 1;
        lut[j * 3] = blue_sums[i];
        lut[j * 3 + 1] = green_sums[i];
        lut[j * 3 + 2] = red_sums[i];
    }

    cv::Mat erodeLabels, dilateLabels;

    cv::Mat kernel = cv::Mat::ones(cv::Size(3, 3), CV_64F);

    cv::Mat labelsCopy = cv::Mat(cv::Size(cols, rows), CV_64FC1);

    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
        {
            labelsCopy.at<double>(i, j) = labels[i*cols + j];
        }
    }

    cv::erode(labelsCopy, erodeLabels, kernel);
    cv::dilate(labelsCopy, dilateLabels, kernel);

    std::vector<double> newLabels(rows * cols);


    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
        {
            if((int)erodeLabels.at<double>(i,j) != (int)labels[i*cols + j] ||
                    (int)dilateLabels.at<double>(i,j) != (int)labels[i*cols + j])
            {
                newLabels[i * cols + j] = 1;
            }
            else
            {
                newLabels[i * cols + j] = labels[i*cols + j] + 1;
            }
            newLabels[i * cols + j] = std::max(1., std::min(newLabels[i * cols + j],
                                                                (double) size));
        }
    }



    cv::Mat bgrNew = cv::Mat::zeros(bgr.size(), CV_8UC3);

    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
        {
            k = newLabels[i * cols + j] - 1;
            bgrNew.at<cv::Vec3b>(i,j)[0] = lut[k * 3] * 255;
            bgrNew.at<cv::Vec3b>(i,j)[1] = lut[k * 3 + 1] * 255;
            bgrNew.at<cv::Vec3b>(i,j)[2] = lut[k * 3 + 2] * 255;
        }
    }

    return bgrNew;
}


int RgbSegmentation::maxVal(const std::vector<int>& vec)
{
    int maxV = -1000;
    int i, end = pixelNumber;
    for(i = 0; i < end; ++i) {
        if(vec[i] > maxV)
            maxV = vec[i];
    }
    return maxV;
}

int RgbSegmentation::findVal(const std::vector<int>& vec, const int &elem, const int &start)
{
    int idx = -1000;
    int i;
    for( i = start; i < pixelNumber; ++i)
    {
        if(elem == vec[i])
            return i;
    }

    return idx;
}

std::vector<double> RgbSegmentation::accumarray(const std::vector<int>& subs, const std::vector<double>& vals)
{
    int i;

    labelSize = maxVal(subs);
    std::vector<double> out(labelSize);

    AccumElem elements[labelSize];

    for(i = 0; i < pixelNumber; ++i)
    {
        elements[subs[i] - 1].index++;
        elements[subs[i+1] - 1].sum += vals[i];
    }

    for(i = 0; i < labelSize; ++i)
    {
        out[i] = elements[i].sum / (double)(elements[i].index);
    }


    return out;
}

std::vector<double> RgbSegmentation::compactLabels(const std::vector<int>& tri_labels, const std::vector<double>& labels)
{

    std::vector<double> newLabels = labels;
    std::vector<int> labelVectA, labelVectC, labelVectIc;

    int i, j, k;
    int sizeLabels = -1;
    for(i = 0; i < sizeTriangles; ++i)
    {
      if(tri_labels[i] > sizeLabels)
      {
        sizeLabels = tri_labels[i];
      }
    }

    bool checked[sizeLabels + 1];
    std::fill_n(checked, sizeLabels + 1, 0);
    int labelArrayC[sizeLabels + 1];
    std::fill_n(labelArrayC, sizeLabels + 1, -1);

    int labelArrayA[pixelNumber];

    k = 0;

    for(j = 0; j < cols; ++j) {
        for(i = 0; i < rows; ++i) {
            int idx = labels[i*cols + j] - 1;
            int elem = tri_labels[idx];
            if(!*(checked + elem)) {
                *(checked + elem) = true;
            }
            *(labelArrayA + k) = elem;
            ++k;
        }
    }

    int counter = 0;
    for(i = 0; i < sizeLabels; ++i) {
        if(checked[i]) {
            *(labelArrayC + i) = counter;
            counter++;
        }
    }

    int labelArrayIc[pixelNumber];

    for(i = 0; i < pixelNumber; i++) {
        *(labelArrayIc + i) = *(labelArrayC + *(labelArrayA + i) );
    }

    i = 0, j = 0;
    for(k = 0; k < pixelNumber && j < cols; k++)
    {
        newLabels[i*cols + j] = *(labelArrayIc + k) + 1;
        ++i;
        if(i == rows) {
            i = 0;
            ++j;
        }
    }

    return newLabels;
}

std::vector<int> RgbSegmentation::integrate_merges(const std::vector<int>& old_labels,
                                                   const std::vector<int>& new_labels,
                                                   const double &merge_threshold) {

    std::vector<int> lut(sizeTriangles), tail(sizeTriangles), next(sizeTriangles);

    for(int i = 0; i < sizeTriangles; ++i)
    {
        lut[i] = tail[i] = i + 1;
    }


    int size = sizeTriangles - 1;
    int old_label, new_label;
    for(int i = 0; i < size; ++i)
    {
        if(merge_costs[i] < merge_threshold)
        {
            old_label = old_labels[i];
            new_label = new_labels[i];
            next[tail[new_label - 1] - 1] = old_label;
            tail[new_label - 1] = tail[old_label - 1];
            tail[old_label - 1] = 0;
        }
    }

    for(int i = 0; i < sizeTriangles; ++i)
    {
        if(tail[i])
        {
            int idx = i + 1;
            while(idx)
            {
                lut[idx - 1] = i + 1;
                idx = next[idx - 1];
            }
        }
    }

    return lut;
}

cv::Mat RgbSegmentation::mat2cvMat(const std::vector<double>& rgb) {
    int i, j;
    int doublePixelNumber = pixelNumber*2;
    cv::Mat img = cv::Mat(cv::Size(rows, cols), CV_8UC3);

    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
        {
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(rgb[i*cols + j],
                                                rgb[i*cols + pixelNumber + j],
                                                rgb[i*cols + doublePixelNumber + j]);
        }
    }

    return img;
}

std::vector<double> RgbSegmentation::cvMat2Mat(const cv::Mat &mat) {
    std::vector<double> img(pixelNumber * 3, 0.);
    int i, j;
    int doublePixelNumber = pixelNumber*2;

    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
        {
            img[i*cols + j] = (double)mat.at<cv::Vec3b>(i, j)[0];
            img[i*cols + pixelNumber + j] = (double)mat.at<cv::Vec3b>(i, j)[1];
            img[i*cols + doublePixelNumber + j] = (double)mat.at<cv::Vec3b>(i, j)[2];
        }
    }

    return img;
}

std::vector<double> RgbSegmentation::dtMat2mat(const std::vector<double>& src) {
    std::vector<double> bgr(pixelNumber * 3, 0.);
    int i, j;
    int doublePixelNumber = pixelNumber*2;

    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
        {
            bgr[i*cols + j] = src[i*cols + j]*255.;
            bgr[i*cols + pixelNumber + j] = src[i*cols + pixelNumber + j]*255.;
            bgr[i*cols + doublePixelNumber + j] = src[i*cols + doublePixelNumber + j]*255.;
        }
    }

    return bgr;
}

bool RgbSegmentation::imfilter(const std::vector<double>& src, const std::vector<double>& filter,
                             std::vector<double>& _mat, const bool &isTranspose, const int &plane) {

    int factor = pixelNumber*plane;
    int i, j, k;
    std::vector<double> vec;
    double sum;

    if( isTranspose ) {
        vec.resize(cols + 15, 0.0);
        for( i=0; i<rows; ++i )
        {
            for(j=0; j<cols; ++j )
                vec[j + 7] = src[i*cols + factor + j];
            for( j=0; j<7; ++j )
                vec[j] = src[i*cols + factor];
            for( j=0; j<7; ++j )
               vec[j + cols + 7] = src[i*cols + factor + cols - 1];

            for( j=0; j<cols; ++j )
            {
                sum = 0.0;
                for( k=0; k<15; ++k )
                {
                    sum += vec[j + k] * (filter[14 - k]);
                }
                _mat[i*cols + factor + j] = sum;
            }
        }
    }
    else
    {
        vec.resize(rows+15, 0.0);
        for( i=0; i < cols; ++i )
        {
            for( j=0; j<rows; ++j )
                vec[j + 7] = src[j*cols + factor + i];
            for( j=0; j<7; ++j )
                vec[j] = src[factor + i];
            for( j=0; j<7; ++j )
                vec[j + rows + 7] = src[(rows - 1)*cols + factor + i];
            for( j=0; j<rows; ++j ) {
                sum = 0.0;
                for( int k=0; k<15; ++k ) {
                    sum += vec[j + k] * filter[14 - k];
                }
                _mat[j*cols + factor + i] = sum;
            }
        }
    }
    return true;
}

bool RgbSegmentation::im2single( const cv::Mat &src, std::vector<double>& mat, const int& plane ) {

    if( src.channels() != 1 )
    {
        return false;
    }

    unsigned char *img = src.data;

    double _eps = 1.0e-10;

    double max_range = 1.0;
    double half_range = 0.0;
    if( src.depth() == CV_8U ) {
        max_range = 255.;
    } else if ( src.depth() == CV_8S ) {
        max_range = 255.;
        half_range = 128.;
    } else if( src.depth() == CV_16U ) {
        max_range = 65535.;
    } else if (src.depth() == CV_16S ) {
        max_range = 65535.;
        half_range = 32768.;
    }

    int factor = pixelNumber*plane;
    double denominator = max_range + _eps;

    for( int i=0; i<rows; ++i)
    {
        for( int j=0; j<cols; ++j)
        {
            mat[i*cols + factor + j] = (( img[i*cols + j] + half_range )
                                            / denominator);
        }
    }

    return true;
}

void RgbSegmentation::rgbtolab()
{

    int i, j;
    mean_color_lab = std::vector<double>(sizeTriangles*3, 0.);//allocateMatrix<double>(sizeTriangles, 3, true, 0.);
    int size = 3;
    double rgb[3], XYZ[3];
    double tmp_value;
    uint offset = 0;

    for(i = 0; i < sizeTriangles; ++i)
    {
        for(j = 0; j < size; ++j)
        {
            tmp_value = mean_color_rgb[offset + j] / 255.;

            (tmp_value > 0.0404482362771076) ? tmp_value = std::pow(((tmp_value + 0.055) / 1.055), 2.4) :
                    tmp_value /= 12.92;

            *(rgb + j) = tmp_value;

        }

        double X, Y, Z;

        X = *rgb * 0.436052025 + *(rgb + 1) * 0.385081593 + *(rgb + 2) * 0.143087414;
        Y = *rgb * 0.222491598 + *(rgb + 1) * 0.716886060 + *(rgb + 2) * 0.060621486;
        Z = *rgb * 0.013929122 + *(rgb + 1) * 0.097097002 + *(rgb + 2) * 0.714185470;


        *XYZ =  X / 0.964296 ;
        *(XYZ + 1) = Y;
        *(XYZ + 2) = Z / 0.825106 ;


        for(j = 0; j < size; ++j)
        {

            tmp_value = *(XYZ + j);

            tmp_value = (tmp_value > 0.008856452) ?  std::pow(tmp_value, 0.333333333333333) :
                     (tmp_value * 7.787037037) + (0.137931034);

            *(XYZ + j) = tmp_value;

        }


        mean_color_lab[offset] = (116. * *(XYZ + 1)) - 16.;
        mean_color_lab[offset + 1] = 500. * (*XYZ - *(XYZ + 1));
        mean_color_lab[offset + 2] = 200. * (*(XYZ + 1) - *(XYZ + 2));
        offset += 3;
    }
}

void RgbSegmentation::computeSquaredEdgeDistances(const double &sigma_c) {

    int size = 3, k, i, j;


    std::vector<double> dc2 = std::vector<double>(sizeTriangles*size, 0.);//allocateMatrix<double>(sizeTriangles, size, true, 0.);
    color_weights = std::vector<double>(sizeTriangles*size, 0.);//allocateMatrix<double>(sizeTriangles, size, false, 0.);
    int currentIdx;
    double delta;
    int offset;

    for(k = 0; k < size; k++)
    {
        offset = 0;
        for(i = 0; i < sizeTriangles; i++)
        {
            for(j = 0; j < size; j++)
            {
                if(!std::isnan(neighbors[offset + j]))
                {
                    currentIdx = neighbors[offset + j] - 1;
                    delta = mean_color_rgb[currentIdx*size + k] -
                            mean_color_rgb[offset + k];
                    dc2[i*size + j] += square<double>(delta);
                }
                else
                {
                    dc2[i*size + j] = 0.;
                }
            }
            offset += size;
        }
    }

    double elem = 1./(sigma_c*sigma_c);

    offset = 0;
    for(i = 0; i < sizeTriangles; i++)
    {
        for(j = 0; j < size; j++)
        {
            color_weights[offset + j] = 1. - (std::exp(-elem * dc2[i*size + j]));
        }
        offset += size;
    }
}


void RgbSegmentation::mxCircleStats(const std::vector<double>& radii, const std::vector<cv::Point2d>& centers,
                                    const std::vector<double>& RGBb) {

    const int ndims = 3; //CHANGE FOR BLACK AND WHITE

    mean_color_rgb = std::vector<double>(sizeTriangles*3, 0.);

    int min_col, max_col, min_row, max_row, count, row, col, j, k;
    double radius, dx, dy;
    Point<double> current_center;
    uint offset = 0;

    for(int i = 0; i < sizeTriangles; i++)
    {

        current_center.x = centers[i].x - 1;
        current_center.y = centers[i].y - 1;

        radius = radii[i];

        min_col = (int)(current_center.x - radius);
        if (min_col < 0) min_col = 0;


        max_col = (int)(current_center.x + radius);
        if (max_col >= cols) max_col = cols - 1;

        min_row = (int)(current_center.y - radius);
        if (min_row < 0) min_row = 0;

        max_row = (int)(current_center.y + radius);
        if (max_row >= rows) max_row = rows - 1;

        radius = square<double>(radius) + 0.25;

        count = 0;

        for (row = min_row; row <= max_row; ++row)
        {
            for (col = min_col; col <= max_col; ++col) {
                dx = col - current_center.x;
                dy = row - current_center.y;

                if ( (dx*dx + dy*dy) < radius ) {

                    for (j=0, k = 2; j < ndims, k >= 0; ++j, k--)
                    {
                        mean_color_rgb[offset + j] += RGBb[row*cols + k*pixelNumber + col];
                    }

                    ++count;
                }
            }
        }

        for (j=0; j < ndims; ++j)
        {
            mean_color_rgb[i*3 + j] /= count;
        }
        offset += ndims;
    }
}


void RgbSegmentation::quicksort(std::vector<cv::Point2i> &XYpoints, int p, int r)
{
    if ( p < r ) {
        int j = partition(XYpoints, p, r);
        quicksort(XYpoints, p, j-1);
        quicksort(XYpoints, j+1, r);
    }
}

int RgbSegmentation::partition(std::vector<cv::Point2i> &XYpoints, int p, int r) {

    int pivot = XYpoints[r].x;

    while ( p < r ) {

        while ( XYpoints[p].x  < pivot ) {
            p++;
        }

        while (XYpoints[r].x > pivot ) {
            r--;
        }

        if (XYpoints[p].x == XYpoints[r].x )
            p++;
        else if ( p < r ) {
            cv::Point2i temp = XYpoints[p];
            XYpoints[p] = XYpoints[r];
            XYpoints[r] = temp;
        }
    }

    return r;
}


void RgbSegmentation::computeCircumcircleOverlap(const std::vector<double>& radii, const std::vector<cv::Point2d>& centers,
                                                 const std::vector<double>& neighbors) {



       edge_lengths = std::vector<double>(sizeTriangles*3, 0.);

       double radii_;
       double r, R, c1, c2;
       double distance;
       double theta1, theta2, s1, s2;
       int idx;
       int offset;

       for(int i = 0; i < 3; i++) {

           offset = 0;

           for(int j = 0; j < sizeTriangles; ++j)
           {
               idx = std::isnan(neighbors[j*3 + i]) ? 0 : neighbors[offset + i] - 1;

               radii_ = radii[idx];

               r = std::min(radii[j], radii_);
               R = std::max(radii[j], radii_);


               const cv::Point2d& p1 = centers[j];
               const cv::Point2d& p2 = centers[idx];

               distance = distPoint2Point<double>(p1.x, p1.y, p2.x, p2.y);


               if(distance == 0.)
               {
                   edge_lengths[offset + i] = 2 * (double)R;
               }
               else
               {
                   const double& squareR = square<double>(R);
                   const double& squareDistance = square<double>(distance);
                   const double& squarer = square<double>(r);
                   const double& denominator = 2 * R * distance;
                   c1 = ((squareR + squareDistance) -
                                 squarer ) / denominator ;
                   c2 = ( (squarer + squareDistance) -
                                 squareR) / squareDistance;

                   c1 = std::max((double)-1., std::min((double)c1, (double)1.));
                   c2 = std::max((double)-1., std::min((double)c2, (double)1.));

                   theta1 = std::acos(c1);
                   theta2 = std::acos(c2);

                   s1 = std::sqrt(1 - square<double>(c1));
                   s2 = std::sqrt(1 - square<double>(c2));

                   edge_lengths[offset + i] = 2*(s1*R);
               }
               offset += 3;
           }
       }
}
