// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess_algorithm_h/indoor_prop_yolov8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>
namespace sweeper_ai
{

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(float* tensor, int dfl_len, float* box)
{
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold,
                      int8_t *score_tensor2, int32_t score_zp2, float score_scale2,
                      std::vector<float> &objProbs2, 
                      std::vector<int> &classId2)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int offset2 = i* grid_w + j;
            int max_class_id = -1;
            int max_class_id2 = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            int8_t max_score2 = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM_INDOOR_YOLOV8; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            if(max_class_id > -1)
            {
                for (int c2 = 0; c2 < DIR_CLASS_NUM_INDOOR_YOLOV8; c2++)
                {
                    if ((score_tensor2[offset2] > score_thres_i8) && (score_tensor2[offset2] > max_score2))
                    {
                        max_score2 = score_tensor2[offset2];
                        max_class_id2 = c2;
                    }
                    offset2 += grid_len;
                }
            }

            // compute box
            if (max_score> score_thres_i8)
            {
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++)
                {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                objProbs2.push_back(deqnt_affine_to_f32(max_score2, score_zp2, score_scale2));
                classId2.push_back(max_class_id2);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int> &classId, 
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM_INDOOR_YOLOV8; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

int indoor_prop_yolov8_post_process(EcoRknnModelParams& modelparams, rknn_output *_outputs, letterbox_t *letter_box, float conf_threshold, 
    float nms_threshold, std::vector<detect_result_t> &proposals)
{
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int>   classId;
    std::vector<float> objProbs2;
    std::vector<int>   classId2;
    int validCount = 0;
    int stride     = 0;
    int grid_h     = 0;
    int grid_w     = 0;
    int model_in_w = (modelparams.nmodelinputweith_)[0];
    int model_in_h = (modelparams.nmodelinputheight_)[0];

    int dfl_len    = modelparams.nmodeloutputchannel_[0] /4;
    int output_per_branch = modelparams.io_num.n_output / 3;

    std::vector<int> cls_2 = {9, 11, 13};
    std::vector<int> cls = {1, 4, 7};
    std::vector<int> box = {0, 3, 6};

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;

        if (output_per_branch == 5){
            score_sum       = _outputs[i*(output_per_branch-2) + 2].buf;
            score_sum_zp    = modelparams.out_zps[i*(output_per_branch-2) + 2];
            score_sum_scale = modelparams.out_scales[i*(output_per_branch-2) + 2];
        }

        int box_idx    = box[i];
        int score_idx  = cls[i];
        int score_idx2 = cls_2[i];


        grid_h = modelparams.nmodeloutputheight_[box_idx];
        grid_w = modelparams.nmodeloutputweith_[box_idx];
        stride = model_in_h / grid_h;

        if (!_outputs[0].want_float)
        {
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, modelparams.out_zps[box_idx], modelparams.out_scales[box_idx],
                                     (int8_t *)_outputs[score_idx].buf, modelparams.out_zps[score_idx], modelparams.out_scales[score_idx],
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale, grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold,
                                     (int8_t *)_outputs[score_idx2].buf, modelparams.out_zps[score_idx2+1], modelparams.out_scales[score_idx2+1],
                                     objProbs2, classId2);

        }
        else
        {
            validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       filterBoxes, objProbs, classId, conf_threshold);
        }

    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || proposals.size() >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];
        int id2 = classId2[n];
        float obj_conf2 = objProbs2[n];

        detect_result_t outputresult;
        BOX_PROP prop;
        BOX_PROP sub_prop;
        outputresult.box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        outputresult.box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        outputresult.box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        outputresult.box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);

        prop.condidence         = obj_conf;
        prop.name               = id;
        outputresult.prop.push_back(prop);
        sub_prop.condidence     = obj_conf2;   //  附属属性
        sub_prop.name           = id2;
        outputresult.sub_prop.push_back(sub_prop);
        outputresult.issure     = true;
        std::cout << "id = " << id << "  id2 = " << id2 << std::endl;

        proposals.push_back(outputresult);
    }

    return 0;
}

// 零拷贝
// 修改rknn_output *_outputs为rknn_tensor_mem **_outputs，添加want_float参数
// _outputs[].buf改为_outputs[]->virt_addr
int indoor_prop_yolov8_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **_outputs, letterbox_t *letter_box, float conf_threshold, 
    float nms_threshold, std::vector<detect_result_t> &proposals, int want_float)
{
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int>   classId;
    std::vector<float> objProbs2;
    std::vector<int>   classId2;
    int validCount = 0;
    int stride     = 0;
    int grid_h     = 0;
    int grid_w     = 0;
    int model_in_w = (modelparams.nmodelinputweith_)[0];
    int model_in_h = (modelparams.nmodelinputheight_)[0];

    int dfl_len    = modelparams.nmodeloutputchannel_[0] /4;
    int output_per_branch = modelparams.io_num.n_output / 3;

    std::vector<int> cls_2 = {9, 11, 13};
    std::vector<int> cls = {1, 4, 7};
    std::vector<int> box = {0, 3, 6};

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;

        if (output_per_branch == 5){
            score_sum       = _outputs[i*(output_per_branch-2) + 2]->virt_addr;
            score_sum_zp    = modelparams.out_zps[i*(output_per_branch-2) + 2];
            score_sum_scale = modelparams.out_scales[i*(output_per_branch-2) + 2];
        }

        int box_idx    = box[i];
        int score_idx  = cls[i];
        int score_idx2 = cls_2[i];

        grid_h = modelparams.nmodeloutputheight_[box_idx];
        grid_w = modelparams.nmodeloutputweith_[box_idx];
        stride = model_in_h / grid_h;

        if (!want_float)
        {
            validCount += process_i8((int8_t *)_outputs[box_idx]->virt_addr, modelparams.out_zps[box_idx], modelparams.out_scales[box_idx],
                                     (int8_t *)_outputs[score_idx]->virt_addr, modelparams.out_zps[score_idx], modelparams.out_scales[score_idx],
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale, grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold,
                                     (int8_t *)_outputs[score_idx2]->virt_addr, modelparams.out_zps[score_idx2], modelparams.out_scales[score_idx2],
                                     objProbs2, classId2);

        }
        else
        {
            validCount += process_fp32((float *)_outputs[box_idx]->virt_addr, (float *)_outputs[score_idx]->virt_addr, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       filterBoxes, objProbs, classId, conf_threshold);
        }

    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || proposals.size() >= OBJ_NUMB_MAX_SIZE)
        {   
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];
        int id2 = classId2[n];
        float obj_conf2 = objProbs2[n];

        detect_result_t outputresult;
        BOX_PROP prop;
        BOX_PROP sub_prop;
        outputresult.box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        outputresult.box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        outputresult.box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        outputresult.box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);

        prop.condidence         = obj_conf;
        prop.name               = id;
        outputresult.prop.push_back(prop);
        sub_prop.condidence     = obj_conf2;   //  附属属性
        sub_prop.name           = id2;
        outputresult.sub_prop.push_back(sub_prop);
        outputresult.issure     = true;
        std::cout << "id = " << id << "  id2 = " << id2 << std::endl;
        std::cout << "prop = " << obj_conf << "  sub_prop = " << obj_conf2 << std::endl;

        proposals.push_back(outputresult);
    }

    return 0;
}

}
