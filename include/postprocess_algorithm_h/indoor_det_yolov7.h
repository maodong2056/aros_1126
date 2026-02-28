#ifndef _RKNN_YOLOV5_INDOOR_POSTPROCESS_H_
#define _RKNN_YOLOV5_INDOOR_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "eco_common.h"

namespace sweeper_ai
{
    int post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, float conf_threshold, 
    float nms_threshold, std::vector<detect_result_t> &proposals);
}
#endif //_RKNN_YOLOV5_INDOOR_POSTPROCESS_H_