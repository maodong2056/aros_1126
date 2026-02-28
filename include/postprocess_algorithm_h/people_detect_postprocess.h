#ifndef ECO_PEOPLE_YOLOX_H_
#define ECO_PEOPLE_YOLOX_H_

#include "opencv2/core/core.hpp"
#include "utils.h"
#include "rknn_api.h"
#include "common.h"


namespace sweeper_ai
{

typedef struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
}GridAndStride;

// typedef struct BBox 
// {
//     float xmin;
//     float ymin;
//     float xmax;
//     float ymax;
//     int label;
//     float score;
// } BBox;

typedef struct BodyBox{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float label;
        float score;
        float btag;
        float undistored; //是否去畸变
    }BodyBox;
    
typedef struct FaceBox{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float label;
    float score;
    float ftag;
    float kps[15];
    float undistored;
}FaceBox;

typedef struct detHuman
{
    std::vector<BodyBox> bbox;
    std::vector<FaceBox> fbox;
} detHuman;

class YoloxModel
{

public:
    YoloxModel(){};
    ~YoloxModel(){};
    
    
    // std::vector<BBox> yolox_postprocess(const std::vector<rknn_output> outputdata, int ori_h, int ori_w, float score_threshold_x, float nms_threshold_,float ratio);
    std::vector<BBox> yolox_postprocess(rknn_output* outputdata, int ori_h, int ori_w, float score_threshold_x, float nms_threshold_,float ratio);
    
    // 零拷贝 人形后处理
    std::vector<BBox> yolox_postprocess_zero_copy(EcoRknnModelParams& modelparams, int8_t *output, int ori_h, int ori_w, float score_threshold_x, float nms_threshold_,float ratio);
    
    void humandet_boxes_filter(std::vector<BodyBox> &bodyboxes, float threshold);

private:
    void generate_grids_and_stride(const int target_size_x, const int target_size_y, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);

    /*num cls = */
    void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const int last_channel_size, 
                const float* feat_ptr, float prob_threshold, std::vector<BBox>& objects);

    void generate_yolox_proposals_zero_copy(EcoRknnModelParams& modelparams, std::vector<GridAndStride> grid_strides, const int last_channel_size, 
            int8_t* feat_ptr, float prob_threshold, std::vector<BBox>& objects);
    
    void nms(std::vector<BBox>& input_boxes, float nms_thresh);
    
    BBox box_transform(BBox ori, int nnh, int nnw, int orih, int oriw,float ratio);

    

    int nnInputHeight_ = 384;
    int nnInputWidth_ = 512;
    std::vector<int> strides_ = { 8, 16, 32 };

    bool eval_ = false;


};
}

#endif