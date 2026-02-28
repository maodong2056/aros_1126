//
// Created by maxiaoyue on 25-8-12.
//

#ifndef RKNN_YOLOV5_DEMO_ANIMAL_DETECT_YOLOV11_POSTPROCESS_H
#define RKNN_YOLOV5_DEMO_ANIMAL_DETECT_YOLOV11_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "eco_common.h"

namespace sweeper_ai
{
    int animal_yolov11_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, scale_t *scale_wh, float conf_threshold, 
    float nms_threshold, std::vector<detect_result_t> &proposals);
}

#endif //RKNN_YOLOV5_DEMO_ANIMAL_DETECT_YOLOV11_POSTPROCESS_H
