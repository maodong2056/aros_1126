#ifndef _RKNN_ECO_FREESPACE_H_
#define _RKNN_ECO_FREESPACE_H_
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <string>
#include <tuple>
#include "common.h"

#define BASE_INPUT_SIZE_WEIGH 512.00f
#define BASE_INPUT_SIZE_HEIGHT 384.00f

#define BASE_REG_MAX 16
// #define USE_NEON 0
#define NEON_VECTOR_SIZE 4
namespace sweeper_ai
{

typedef struct Pointf{
    float x;
    float y;
    float label;
    float id;
    float flag; 
    float confidence;
}Pointf;

typedef struct result_counter{
    std::vector<Pointf> ps;
    float id;
}result_counter;


void gfl_post_process(std::vector<BBox>& results, rknn_output *outputdata, int num_class, int outputdata_size, int nn_in_h, int nn_in_w, int imgsrc_h, int imgsrc_w, float score_threshold, float nms_thres, int **outputsize_weith, int **outputsize_height);

void find_classes_contours(cv::Mat &mask, result_counter& p1, std::vector<BBox>& result_boxes, uint8_t *data, int h, int w, int input_h, int input_w, int org_h, int org_w, int& cm_distance); // 获取所需类别轮廓

// 零拷贝 增加int2float
void find_classes_contours_zero_copy(cv::Mat &mask, result_counter& p1, std::vector<BBox>& result_boxes, int8_t *data, int h, int w, int input_h, int input_w, int org_h, int org_w, int& cm_distance, int32_t zp, float scale); // 获取所需类别轮廓

// 零拷贝
void find_classes_contours_fast_zero_copy(cv::Mat &mask, result_counter& p1, std::vector<BBox>& result_boxes, int8_t *data, int h, int w, int input_h, int input_w, int org_h, int org_w, int& cm_distance); // 获取所需类别轮廓

std::map<int, Pointf> extractBottomContourFromPoints(const std::vector<Pointf>& points);

}
#endif