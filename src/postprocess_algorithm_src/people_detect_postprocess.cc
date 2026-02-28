#include <iostream>
#include "postprocess_algorithm_h/people_detect_postprocess.h"
#include "utils.h"

namespace sweeper_ai
{

void YoloxModel::generate_grids_and_stride(const int target_size_x, const int target_size_y, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides){
     
    for (auto stride : strides)
    {
        int num_grid_x = target_size_x / stride;
        int num_grid_y = target_size_y / stride;
        // printf("grid %d %d\n", num_grid_x, num_grid_y);
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back(GridAndStride{ g0, g1, stride });
            }
        }
    }

}

/*num cls = */
void YoloxModel::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const int last_channel_size, const float* feat_ptr, float prob_threshold, std::vector<BBox>& objects){
    
     
    const int num_class = last_channel_size - 5;
    // std::cout <<" " <<num_class << std::endl;
    const int num_anchors = grid_strides.size();

    //    const float* feat_ptr = (float *)feat_blob.data;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        float box_objectness = feat_ptr[anchor_idx * last_channel_size + 4];

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
           
            float box_cls_score = feat_ptr[anchor_idx * last_channel_size + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                const int grid0 = grid_strides[anchor_idx].grid0;
                const int grid1 = grid_strides[anchor_idx].grid1;
                const int stride = grid_strides[anchor_idx].stride;

                float x_center = (feat_ptr[anchor_idx * last_channel_size] + grid0) * stride;
                float y_center = (feat_ptr[anchor_idx * last_channel_size + 1] + grid1) * stride;
                float w = exp(feat_ptr[anchor_idx * last_channel_size + 2]) * stride;
                float h = exp(feat_ptr[anchor_idx * last_channel_size + 3]) * stride;
                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;
                objects.push_back(BBox{x0, y0, w+x0, h+y0, float(class_idx), box_prob});

            }
        }
    } //

}

// 零拷贝 int2float
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

void YoloxModel::generate_yolox_proposals_zero_copy(EcoRknnModelParams& modelparams, std::vector<GridAndStride> grid_strides, const int last_channel_size, int8_t* feat_ptr, float prob_threshold, std::vector<BBox>& objects){
    int32_t zp = modelparams.out_zps[0];
    float scale = modelparams.out_scales[0];
     
    const int num_class = last_channel_size - 5;
    // std::cout <<" " <<num_class << std::endl;
    const int num_anchors = grid_strides.size();

    //    const float* feat_ptr = (float *)feat_blob.data;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        float box_objectness = deqnt_affine_to_f32(feat_ptr[anchor_idx * last_channel_size + 4], zp, scale);

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
           
            float box_cls_score = deqnt_affine_to_f32(feat_ptr[anchor_idx * last_channel_size + 5 + class_idx], zp, scale);
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                const int grid0 = grid_strides[anchor_idx].grid0;
                const int grid1 = grid_strides[anchor_idx].grid1;
                const int stride = grid_strides[anchor_idx].stride;

                float x_center = (deqnt_affine_to_f32(feat_ptr[anchor_idx * last_channel_size], zp, scale) + grid0) * stride;
                float y_center = (deqnt_affine_to_f32(feat_ptr[anchor_idx * last_channel_size + 1], zp, scale) + grid1) * stride;
                float w = exp(deqnt_affine_to_f32(feat_ptr[anchor_idx * last_channel_size + 2], zp, scale)) * stride;
                float h = exp(deqnt_affine_to_f32(feat_ptr[anchor_idx * last_channel_size + 3], zp, scale)) * stride;
                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;
                objects.push_back(BBox{x0, y0, w+x0, h+y0, float(class_idx), box_prob});

            }
        }
    } //

}


std::vector<BBox> YoloxModel::yolox_postprocess(rknn_output* outputdata,  int ori_h, int ori_w, float score_threshold_x, float nms_threshold_, float ratio)
{
    // RK output data 
    float* output_float =  (float*)outputdata[0].buf;
    // int last_channel_size = outputdata[0].data_shape.d[2];//[1, anchor_indexes, box + cls_scores]
    int last_channel_size = 6;//[1, anchor_indexes, box + cls_scores]

    std::vector<GridAndStride> grid_strides;
    std::vector<BBox> results;

    this->generate_grids_and_stride(nnInputWidth_, nnInputHeight_, strides_, grid_strides);
    
    this->generate_yolox_proposals(grid_strides, last_channel_size, output_float, score_threshold_x, results);

    std::sort(results.begin(), results.end(), [](BBox a, BBox b) { return a.score > b.score; });

    std::vector<BBox> final_boxes;
    for (size_t i = 0; i < int(results.size()); i++)
    {
        final_boxes.push_back(box_transform(results[i], nnInputHeight_, nnInputWidth_, ori_h, ori_w, ratio ));
    }

    if (!eval_)
    {
        nms(final_boxes, nms_threshold_);
    }
    
    return final_boxes;
}

// 零拷贝
std::vector<BBox> YoloxModel::yolox_postprocess_zero_copy(EcoRknnModelParams& modelparams, int8_t *output,  int ori_h, int ori_w, float score_threshold_x, float nms_threshold_, float ratio)
{
    // RK output data 
    // float* output_float =  (float*)outputdata[0].buf;
    // int last_channel_size = outputdata[0].data_shape.d[2];//[1, anchor_indexes, box + cls_scores]
    int last_channel_size = 6;//[1, anchor_indexes, box + cls_scores]

    std::vector<GridAndStride> grid_strides;
    std::vector<BBox> results;

    this->generate_grids_and_stride(nnInputWidth_, nnInputHeight_, strides_, grid_strides);
    
    // this->generate_yolox_proposals(grid_strides, last_channel_size, output, score_threshold_x, results);
    this->generate_yolox_proposals_zero_copy(modelparams, grid_strides, last_channel_size, output, score_threshold_x, results);

    std::sort(results.begin(), results.end(), [](BBox a, BBox b) { return a.score > b.score; });

    std::vector<BBox> final_boxes;
    for (size_t i = 0; i < int(results.size()); i++)
    {
        final_boxes.push_back(box_transform(results[i], nnInputHeight_, nnInputWidth_, ori_h, ori_w, ratio ));
    }

    if (!eval_)
    {
        nms(final_boxes, nms_threshold_);
    }
    
    return final_boxes;
}

BBox YoloxModel::box_transform(BBox ori, int nn_h, int nn_w, int ori_h, int ori_w,float ratio)
{

    float xmin_origin  = ori.xmin /ratio;
    float ymin_origin  = ori.ymin /ratio;
    float xmax_origin  = ori.xmax /ratio;
    float ymax_oringin = ori.ymax /ratio;

    xmin_origin  = std::max(xmin_origin,  0.f);
    ymin_origin  = std::max(ymin_origin,  0.0f);
    xmax_origin  = std::min(xmax_origin,  ori_w - 1.0f);
    ymax_oringin = std::min(ymax_oringin, ori_w - 1.0f);

    return BBox{xmin_origin, ymin_origin, 
                    xmax_origin, 
                    ymax_oringin,
                    ori.label,
                    ori.score };
}


void YoloxModel::nms(std::vector<BBox>& input_boxes, float nms_thresh)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BBox a, BBox b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) 
    {
        vArea[i] = (input_boxes.at(i).xmax - input_boxes.at(i).xmin + 1)
            * (input_boxes.at(i).ymax - input_boxes.at(i).ymin + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) 
    {
        for (int j = i + 1; j < int(input_boxes.size());) 
        {
            float xx1 = (std::max)(input_boxes[i].xmin, input_boxes[j].xmin);
            float yy1 = (std::max)(input_boxes[i].ymin, input_boxes[j].ymin);
            float xx2 = (std::min)(input_boxes[i].xmax, input_boxes[j].xmax);
            float yy2 = (std::min)(input_boxes[i].ymax, input_boxes[j].ymax);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_thresh) 
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else 
            {
                j++;
            }
        }
    }
}


void YoloxModel::humandet_boxes_filter(std::vector<BodyBox> &bodyboxes, float threshold){
    for(int i = 0;i < bodyboxes.size(); i++){
		BodyBox* a = &bodyboxes[i];
		if(a->label == -1)
            continue; 
		for(int j = i+1;j< bodyboxes.size();j++){
			BodyBox* b = &bodyboxes[j];
			if(b->label == -1)
                continue; 
            
			if(a->label == b->label){
                float iou = 0.f;
                // printf("(%f,%f)(%f,%f),%f",a->xmin,a->ymin,a->xmax, a->ymax,a->label);
                // printf("(%f,%f)(%f,%f),%f",b->xmin,b->ymin,b->xmax, b->ymax,b->label);
				if (a->xmin > b->xmax || a->xmax < b->xmin || a->ymin > b->ymax || a->ymax < b->ymin)
	            {
		            /* no intersection */
		           iou = 0.f;

	            }
                else{
                    float inter_width = std::min(a->xmax, b->xmax) - std::max(a->xmin, b->xmin);
                    float inter_height = std::min(a->ymax, b->ymax) - std::max(a->ymin, b->ymin);
                    float inter_area = inter_width * inter_height;
                    float union_area =  (a->xmax - a->xmin) * (a->ymax - a->ymin) +  (b->xmax - b->xmin) * (b->ymax - b->ymin) - inter_area;
                    iou = inter_area / (union_area + 1) ;
                }
                
				if(iou >= threshold){
					b->label = -1;
#if 1
					printf("human det nms iou %f\n",iou);
#endif
				}
                else{
                    // printf("nms iou %f pass\n",iou);
                }
			}
			else{
				;
			}
		}
	}
}
}
