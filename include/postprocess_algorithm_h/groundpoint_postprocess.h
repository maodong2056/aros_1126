#ifndef _RKNN_GROUNDPOINT_POSTPROCESS_H_
#define _RKNN_GROUNDPOINT_POSTPROCESS_H_
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <string>
#include <tuple>
#include "common.h"

#define GROUNDPOINT_NUMB_MAX_SIZE 256  // 一行最多有64个点

namespace sweeper_ai
{
    int groundpoint_det_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance);
    int groundSemantic_det_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance);
    int groundLine_det_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance);

    // 零拷贝
    int groundpoint_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance);

    int groundLine_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance);

    int groundSemantic_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance);
}
#endif