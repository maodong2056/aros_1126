#ifndef _RKNN_YOLOV5_LIQUID_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_LIQUID_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "eco_common.h"

namespace sweeper_ai
{
    // 零拷贝
    int liquid_post_process_zero_copy_v5(sweeper_ai::EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, scale_t *scale_wh, float conf_threshold, float nms_threshold, std::vector<detect_result_t>& proposals, int want_float);
}
#endif