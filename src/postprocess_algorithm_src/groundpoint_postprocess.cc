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

#include "postprocess_algorithm_h/groundpoint_postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>

#include <set>
#include <vector>

namespace sweeper_ai
{

inline static int clamp(int val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

// sigmoid
static float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
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

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }


static int groundLine_process_i8(int8_t *cls_pred, int8_t *reg_pred, int grid_h, int grid_w, int src_height, int src_width, 
                                        int32_t cls_zp, int32_t reg_zp, float cls_scale, float reg_scale,float threshold,
                                        std::vector<int> &points, std::vector<float> &objProbs, int& cm_distance)
{
    int validCount        = 0;
    int grid_len          = grid_h * grid_w;
    float unsig_threshold = unsigmoid(threshold);  // unsigmoid(threshold); 
    int8_t thres_i8       = qnt_f32_to_affine(unsig_threshold, cls_zp, cls_scale);
    // float deqnt_score = deqnt_affine_to_f32(cls_pred[0], cls_zp, cls_scale);

    float ratio_h = src_height / grid_h;
    float ratio_w = src_width  / grid_w;
    int   max_h   = cm_distance / ratio_h;

    for (int row = max_h; row < grid_h - 1; row++)   // 解析镜头中线以下
    {
        for (int col = 0; col < grid_w; col++)
        {
            int index =  grid_w * row + col;
            int8_t score = cls_pred[index];
            if (score > thres_i8)
            {
                int8_t offset_x = reg_pred[index];
                int8_t offset_y = reg_pred[grid_len + index];
                float x = (col + deqnt_affine_to_f32(offset_x, reg_zp, reg_scale)) * ratio_w;
                float y = (row + deqnt_affine_to_f32(offset_y, reg_zp, reg_scale)) * ratio_h;
        
                // float deqnt_score = deqnt_affine_to_f32(score, cls_zp, cls_scale);
                float deqnt_score = sigmoid(deqnt_affine_to_f32(score, cls_zp, cls_scale));
                points.push_back(x);
                points.push_back(y);
                objProbs.push_back(deqnt_score);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int groundSemantic_process_i8(int8_t *cls_pred, int8_t *reg_pred, int grid_h, int grid_w, int src_height, int src_width, 
                                        int32_t cls_zp, int32_t reg_zp, float cls_scale, float reg_scale,float threshold,
                                        std::vector<int> &points, std::vector<float> &objProbs, int& cm_distance, std::vector<int> &classId)
{
    int validCount        = 0;
    int grid_len          = grid_h * grid_w;
    float unsig_threshold = unsigmoid(threshold);  // unsigmoid(threshold); 
    int8_t thres_i8       = qnt_f32_to_affine(unsig_threshold, cls_zp, cls_scale);
    // float deqnt_score = deqnt_affine_to_f32(cls_pred[0], cls_zp, cls_scale);

    float ratio_h = src_height / grid_h;
    float ratio_w = src_width  / grid_w;
    int   max_h   = cm_distance / ratio_h;

    for (int row = max_h; row < grid_h - 1; row++)   // 解析镜头中线以下
    {
        for (int col = 0; col < grid_w; col++)
        {
            int index =  grid_w * row + col;
            int8_t score = cls_pred[index];


            int8_t max_score = -thres_i8;
            int max_class_id = -1;
            for (int c = 0; c < 3; c++){
                if ((cls_pred[index] > thres_i8) && (cls_pred[index] > max_score))
                {
                    max_score = cls_pred[index];
                    max_class_id = c;
                }
                index += grid_len;
            }

            if (max_score > thres_i8)
            {
                int index =  grid_w * row + col;
                int8_t offset_x = reg_pred[index];
                int8_t offset_y = reg_pred[grid_len + index];
                float x = (col + deqnt_affine_to_f32(offset_x, reg_zp, reg_scale)) * ratio_w;
                float y = (row + deqnt_affine_to_f32(offset_y, reg_zp, reg_scale)) * ratio_h;
        
                // float deqnt_score = deqnt_affine_to_f32(score, cls_zp, cls_scale);
                float deqnt_score = sigmoid(deqnt_affine_to_f32(max_score, cls_zp, cls_scale));
                points.push_back(x);
                points.push_back(y);
                objProbs.push_back(deqnt_score);
                validCount ++;
                classId.push_back(max_class_id);
            }
        }
    }
    return validCount;
}

static int groundpoint_process_i8(int8_t *cls_pred, int8_t *reg_pred, int grid_h, int grid_w, int src_height, int src_width, 
                                        int32_t cls_zp, int32_t reg_zp, float cls_scale, float reg_scale,float threshold,
                                        std::vector<int> &points, std::vector<float> &objProbs, int& cm_distance)
{
    int validCount        = 0;
    int grid_len          = grid_h * grid_w;
    float unsig_threshold = unsigmoid(threshold);  // unsigmoid(threshold); 
    int8_t thres_i8       = qnt_f32_to_affine(unsig_threshold, cls_zp, cls_scale);
    // float deqnt_score = deqnt_affine_to_f32(cls_pred[0], cls_zp, cls_scale);

    float ratio_h = src_height / grid_h;
    float ratio_w = src_width  / grid_w;
    int   max_h   = cm_distance / ratio_h;

    for (int col = 0; col < grid_w; col++)
    {
            int8_t top_score = qnt_f32_to_affine(-1, cls_zp, cls_scale);;
            float x = -1;
            float y = -1;
            float deqnt_score = -1;

            for (int row = max_h; row < grid_h - 1; row++)   // 解析镜头中线以下
            {
                int index =  grid_w * row + col;
                int8_t score = cls_pred[index];
                if ((score > top_score) && (score > thres_i8))
                {
                        int8_t offset_x = reg_pred[index];
                        int8_t offset_y = reg_pred[grid_len + index];
  
                        x = (col + deqnt_affine_to_f32(offset_x, reg_zp, reg_scale)) * ratio_w;
                        y = (row + deqnt_affine_to_f32(offset_y, reg_zp, reg_scale)) * ratio_h;
                        // deqnt_score = deqnt_affine_to_f32(score, cls_zp, cls_scale);
                        deqnt_score = sigmoid(deqnt_affine_to_f32(score, cls_zp, cls_scale));
                        top_score = score;
                }
            }

            if (x == -1){
                x = (col + 0.499561) * ratio_w; 
            }

            points.push_back(x);
            points.push_back(y);
            objProbs.push_back(deqnt_score);
            validCount ++;
    }
    return validCount;
}




static int process_i8(int8_t *cls_pred, int8_t *reg_pred, int grid_h, int grid_w, int src_height, int src_width, 
                                        int32_t cls_zp, int32_t reg_zp, float cls_scale, float reg_scale,float threshold,
                                        std::vector<int> &points, std::vector<float> &objProbs)
{
    int validCount    = 0;
    int grid_len      = grid_h * grid_w;
    float unsig_threshold = unsigmoid(threshold);
    int8_t thres_i8   = qnt_f32_to_affine(unsig_threshold, cls_zp, cls_scale);

    // int8_t thres_i8   = qnt_f32_to_affine(threshold, cls_zp, cls_scale);


    float ratio_h = src_height / grid_h;
    float ratio_w = src_width  / grid_w;

    for (int row = grid_h / 2; row < grid_h - 1; row++)   // 解析镜头中线以下
    {
        for (int col = 0; col < grid_w; col++)
        {
            int index =  grid_w * row + col;
            int8_t score = cls_pred[index];
            if (score > thres_i8)
            {
                int8_t offset_x = reg_pred[index];
                int8_t offset_y = reg_pred[grid_len + index];
                float x = (col + deqnt_affine_to_f32(offset_x, reg_zp, reg_scale)) * ratio_w;
                float y = (row + deqnt_affine_to_f32(offset_y, reg_zp, reg_scale)) * ratio_h;
        
                // float deqnt_score = deqnt_affine_to_f32(score, cls_zp, cls_scale);
                float deqnt_score = sigmoid(deqnt_affine_to_f32(score, cls_zp, cls_scale));
                points.push_back(x);
                points.push_back(y);
                objProbs.push_back(deqnt_score);
                validCount ++;
            }
        }
    }
    return validCount;
}

int groundpoint_det_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance)
{
    std::vector<int> filterPoints;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int cls_num =1;


    cls_num = (modelparams.nmodeloutputchannel_)[0];
    grid_h  = (modelparams.nmodeloutputheight_)[0];
    grid_w  = (modelparams.nmodeloutputweith_)[0];


    if (true)
    {
        validCount += groundpoint_process_i8((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, grid_h, grid_w, src_height, src_width, 
        modelparams.out_zps[0],  modelparams.out_zps[1], modelparams.out_scales[0], modelparams.out_scales[1], conf_threshold, 
        filterPoints, objProbs, cm_distance);                
    } 

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    // std::vector<int> indexArray;
    // for (int i = 0; i < validCount; ++i)
    // {
    //     indexArray.push_back(i);
    // }
    // if (validCount> GROUNDPOINT_NUMB_MAX_SIZE)
    // {
    //     quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    // }
    
    EcoKeyPoint keypoint_single;
    for (int i = 0; i < validCount; ++i)
    {
        // if (indexArray[i] == -1)
        // {
        //     continue;
        // }
        // if (maskdata.size() >= GROUNDPOINT_NUMB_MAX_SIZE)
        // {
        //     break;
        // }
        // int n = indexArray[i];
        keypoint_single.bistrue         = true;
        keypoint_single.mappos.x        = clamp(filterPoints[i * 2 + 0], 0, src_width);
        keypoint_single.mappos.y        = clamp(filterPoints[i * 2 + 1], 0, src_height);
        keypoint_single.fconfidence     = objProbs[i];
        keypoint_single.inlabel         = 7;  // 强行给接地点类别赋值，否则在输出ai_full_result会被过滤掉，uchair:4  地毯：2
        keypoint_single.id              = 0;
        keypoint_single.total_num_curve = 1;
        maskdata.push_back(keypoint_single);
    }
    return 0;
}

// 零拷贝
int groundpoint_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance)
{
    std::vector<int> filterPoints;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int cls_num =1;


    cls_num = (modelparams.nmodeloutputchannel_)[0];
    grid_h  = (modelparams.nmodeloutputheight_)[0];
    grid_w  = (modelparams.nmodeloutputweith_)[0];


    if (true)
    {
        validCount += groundpoint_process_i8((int8_t *)outputs[0]->virt_addr, (int8_t *)outputs[1]->virt_addr, grid_h, grid_w, src_height, src_width, 
        modelparams.out_zps[0],  modelparams.out_zps[1], modelparams.out_scales[0], modelparams.out_scales[1], conf_threshold, 
        filterPoints, objProbs, cm_distance);                
    } 

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    // std::vector<int> indexArray;
    // for (int i = 0; i < validCount; ++i)
    // {
    //     indexArray.push_back(i);
    // }
    // if (validCount> GROUNDPOINT_NUMB_MAX_SIZE)
    // {
    //     quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    // }
    
    EcoKeyPoint keypoint_single;
    for (int i = 0; i < validCount; ++i)
    {
        // if (indexArray[i] == -1)
        // {
        //     continue;
        // }
        // if (maskdata.size() >= GROUNDPOINT_NUMB_MAX_SIZE)
        // {
        //     break;
        // }
        // int n = indexArray[i];
        keypoint_single.bistrue         = true;
        keypoint_single.mappos.x        = clamp(filterPoints[i * 2 + 0], 0, src_width);
        keypoint_single.mappos.y        = clamp(filterPoints[i * 2 + 1], 0, src_height);
        keypoint_single.fconfidence     = objProbs[i];
        keypoint_single.inlabel         = 7;  // 强行给接地点类别赋值，否则在输出ai_full_result会被过滤掉，uchair:4  地毯：2
        keypoint_single.id              = 0;
        keypoint_single.total_num_curve = 1;
        maskdata.push_back(keypoint_single);
    }
    return 0;
}

// yp 修改电线头 4、5
int groundLine_det_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance)
{
    std::vector<int> filterPoints;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int cls_num =1;


    cls_num = (modelparams.nmodeloutputchannel_)[2];
    grid_h  = (modelparams.nmodeloutputheight_)[2];
    grid_w  = (modelparams.nmodeloutputweith_)[2];


    if (true)
    {
        validCount += groundLine_process_i8((int8_t *)outputs[4].buf, (int8_t *)outputs[5].buf, grid_h, grid_w, src_height, src_width, 
        modelparams.out_zps[4],  modelparams.out_zps[5], modelparams.out_scales[4], modelparams.out_scales[5], conf_threshold, 
        filterPoints, objProbs, cm_distance);                
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
    if (validCount> GROUNDPOINT_NUMB_MAX_SIZE)
    {
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    }

    EcoKeyPoint keypoint_single;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }

        if (maskdata.size() >= GROUNDPOINT_NUMB_MAX_SIZE)
        {
            break;
        }
        int n = indexArray[i];
        keypoint_single.bistrue         = true;
        keypoint_single.mappos.x        = clamp(filterPoints[n * 2 + 0], 0, src_width);
        keypoint_single.mappos.y        = clamp(filterPoints[n * 2 + 1], 0, src_height);
        keypoint_single.fconfidence     = objProbs[n];
        keypoint_single.inlabel         = 8;
        keypoint_single.id              = 0;
        keypoint_single.total_num_curve = 1;
        maskdata.push_back(keypoint_single);

    }
    return 0;
}

// 零拷贝
int groundLine_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance)
{
    std::vector<int> filterPoints;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int cls_num =1;


    cls_num = (modelparams.nmodeloutputchannel_)[2];
    grid_h  = (modelparams.nmodeloutputheight_)[2];
    grid_w  = (modelparams.nmodeloutputweith_)[2];


    if (true)
    {
        validCount += groundLine_process_i8((int8_t *)outputs[4]->virt_addr, (int8_t *)outputs[5]->virt_addr, grid_h, grid_w, src_height, src_width, 
        modelparams.out_zps[4],  modelparams.out_zps[5], modelparams.out_scales[4], modelparams.out_scales[5], conf_threshold, 
        filterPoints, objProbs, cm_distance);                
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
    if (validCount> GROUNDPOINT_NUMB_MAX_SIZE)
    {
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    }

    EcoKeyPoint keypoint_single;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }

        if (maskdata.size() >= GROUNDPOINT_NUMB_MAX_SIZE)
        {
            break;
        }
        int n = indexArray[i];
        keypoint_single.bistrue         = true;
        keypoint_single.mappos.x        = clamp(filterPoints[n * 2 + 0], 0, src_width);
        keypoint_single.mappos.y        = clamp(filterPoints[n * 2 + 1], 0, src_height);
        keypoint_single.fconfidence     = objProbs[n];
        keypoint_single.inlabel         = 8;
        keypoint_single.id              = 0;
        keypoint_single.total_num_curve = 1;
        maskdata.push_back(keypoint_single);

    }
    return 0;
}

// yp 新增地毯头 2、3
int groundSemantic_det_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance)
{
    std::vector<int> filterPoints;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int cls_num =1;


    cls_num = (modelparams.nmodeloutputchannel_)[2];
    grid_h  = (modelparams.nmodeloutputheight_)[2];
    grid_w  = (modelparams.nmodeloutputweith_)[2];


    if (true)
    {
        validCount += groundLine_process_i8((int8_t *)outputs[2].buf, (int8_t *)outputs[3].buf, grid_h, grid_w, src_height, src_width, 
        modelparams.out_zps[2],  modelparams.out_zps[3], modelparams.out_scales[2], modelparams.out_scales[3], conf_threshold, 
        filterPoints, objProbs, cm_distance);                
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
    if (validCount> GROUNDPOINT_NUMB_MAX_SIZE)
    {
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    }

    EcoKeyPoint keypoint_single;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }

        if (maskdata.size() >= 2 * GROUNDPOINT_NUMB_MAX_SIZE)
        {
            break;
        }
        int n = indexArray[i];
        keypoint_single.bistrue         = true;
        keypoint_single.mappos.x        = clamp(filterPoints[n * 2 + 0], 0, src_width);
        keypoint_single.mappos.y        = clamp(filterPoints[n * 2 + 1], 0, src_height);
        keypoint_single.fconfidence     = objProbs[n];
        keypoint_single.inlabel         = 9;
        keypoint_single.id              = 0;
        keypoint_single.total_num_curve = 1;
        maskdata.push_back(keypoint_single);

    }
    return 0;
}

// 零拷贝 提线输出 0 地毯边 1 流苏边 2 门槛边
int groundSemantic_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, float conf_threshold, int src_height,  int src_width,  std::vector<EcoKeyPoint>& maskdata, int& cm_distance)
{
    std::vector<int> filterPoints;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int cls_num =1;


    cls_num = (modelparams.nmodeloutputchannel_)[2];
    grid_h  = (modelparams.nmodeloutputheight_)[2];
    grid_w  = (modelparams.nmodeloutputweith_)[2];


    if (true)
    {
        validCount += groundSemantic_process_i8((int8_t *)outputs[2]->virt_addr, (int8_t *)outputs[3]->virt_addr, grid_h, grid_w, src_height, src_width, 
        modelparams.out_zps[2],  modelparams.out_zps[3], modelparams.out_scales[2], modelparams.out_scales[3], conf_threshold, 
        filterPoints, objProbs, cm_distance, classId);
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
    if (validCount> GROUNDPOINT_NUMB_MAX_SIZE)
    {
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    }

    EcoKeyPoint keypoint_single;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }

        if (maskdata.size() >= GROUNDPOINT_NUMB_MAX_SIZE)
        {
            break;
        }
        int n = indexArray[i];
        keypoint_single.bistrue         = true;
        keypoint_single.mappos.x        = clamp(filterPoints[n * 2 + 0], 0, src_width);
        keypoint_single.mappos.y        = clamp(filterPoints[n * 2 + 1], 0, src_height);
        keypoint_single.fconfidence     = objProbs[n];

        if (classId[i] == 0)
        {
            keypoint_single.inlabel         = 9;
        }
        else if (classId[i] == 1)
        {
            keypoint_single.inlabel         = 10;
        }
        else if (classId[i] == 2)
        {
            keypoint_single.inlabel         = 11;
        }

        keypoint_single.id              = 0;
        keypoint_single.total_num_curve = 1;
        maskdata.push_back(keypoint_single);

    }
    return 0;
}
}