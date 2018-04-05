# ASML
Skin Lesion Image Segmentation Using Delaunay Triangulation for Melanoma Detection (ASML) is a fast and fully-automatic algorithm for skin lesion segmentation in dermoscopic images. Delaunay Triangulation is used to extract a binary mask of the lesion region, without the need of any training stage. The proposed approach is highly accurate when dealing with benign lesions, while the detection accuracy significantly decreases when melanoma images are segmented.
For more details, you can read the following papers:
1. Melanoma Detection Using Delaunay Triangulation [[1]](./papers/melanoma_detection_using_delaunay_triangulation.pdf)
2. Skin Lesion Image Segmentation Using Delaunay Triangulation for Melanoma Detection [[2]](./papers/skin_lesion_image_segmentation_using_delaunay_triangulation_for_melanoma_detection.pdf)

## Requirements
* CGal
* OpenCV
* Qt5
* Boost

## Results
In order to carry out a quantitative evaluation of the ASLM algorithm, we took into account six well-known segmentation methods, namely JSEG, SRM, KPP, K-means, Otsu, and Level Set. Four different metrics have been selected to calculate the segmentation results: sensitivity, specificity, accuracy, and F-measure. We tested our algorithm on the [PH2 dataset](http://www.fc.up.pt/addi/ph2%20database.html).

| **Method** | **Sensitivity** | **Specificity** | **Accuracy** | **F-measure** |
|--------------|:-------------------:|:--------------------:|:------------------:|:---------------:|
| JSEG | 0.7108 |  0.9714 |  0.8947 ± 0.0176 | 0.7554 |
| SRM | 0.1035 | 0.8757 | 0.6766 ± 0.0346 |  0.1218 |
| KPP |  0.4147 |  0.9581 |  0.7815 ± 0.0356 | 0.5457 |
| K-means |  0.7291 |  0.8430 | 0.8249 ± 0.0107 |  0.6677 |
| Otsu |  0.5221 | 0.7064 | 0.6518 ± 0.0203 |  0.4293 |
| Level Set | 0.7188 | 0.8003 |  0.7842 ± 0.0295 | 0.6456 |
| ASML | **0.8024** |  **0.9722** | **0.8966 ± 0.0276** | **0.8257** |

The table shows the segmentation results obtained by considering the complete PH2 data set (200 images). ASLM achieves the best performance with respect to the other considered segmentation algorithms on all the used evaluation metrics. Moreover, the only comparable results on accuracy and specificity are obtained by JSEG. It is important to point out that, in the computation of the experimental measures, JSEG has been used as a semi-automatic method, manually merging, in case of over-segmentation, the correctly detected lesion regions.
ASLM is a fully-automatic method and no adjustments to the generated binary mask have been performed. More details and experimental evaluations are explained in the papers [[1]](./papers/melanoma_detection_using_delaunay_triangulation.pdf) and [[2]](./papers/skin_lesion_image_segmentation_using_delaunay_triangulation_for_melanoma_detection.pdf).

