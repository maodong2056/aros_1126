#ifndef _RKNN_YOLOV8_INDOOR_PROP_POSTPROCESS_H_
#define _RKNN_YOLOV8_INDOOR_PROP_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "eco_common.h"

namespace sweeper_ai
{
    int indoor_prop_yolov8_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, float conf_threshold, 
    float nms_threshold, std::vector<detect_result_t> &proposals);
    // 零拷贝
    int indoor_prop_yolov8_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, letterbox_t *letter_box, float conf_threshold, 
    float nms_threshold, std::vector<detect_result_t> &proposals, int want_float);
}
#endif //_RKNN_YOLOV8_INDOOR_PROP_POSTPROCESS_H_