#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <stdint.h>
#include <eco_task_infer.h>
#include <unordered_set>
#include "postprocess_algorithm_h/common.h"
#include "postprocess_algorithm_h/indoor_det_yolov7.h"
#include "postprocess_algorithm_h/indoor_det_yolov8.h"
#include "postprocess_algorithm_h/indoor_prop_yolov8.h"
#include "postprocess_algorithm_h/freespace_postprocess.h"
#include "postprocess_algorithm_h/people_detect_postprocess.h"
#include "postprocess_algorithm_h/pm_detect_postprocess.h"
#include "postprocess_algorithm_h/groundpoint_postprocess.h"
#include "postprocess_algorithm_h/infrared_detect_postprocess.h"
#include "postprocess_algorithm_h/animal_detect_postprocess.h"
#include "postprocess_algorithm_h/liquid_detect_postprocess.h"
#include "postprocess_algorithm_h/yolov5_liquid_detect_postprocess.h"
#include "postprocess_algorithm_h/animal_detect_yolov11_postprocess.h"
#include "postprocess_algorithm_h/obstacle_detect_postprocess.h"
#include "postprocess_algorithm_h/lint_detect_postprocess.h"
#include "postprocess_algorithm_h/drystain_detect_postprocess.h"
#include "postprocess_algorithm_h/yolov8_irstain_detect_postprocess.h"


namespace sweeper_ai
{

    //结构光数据辅助AI地毯识别
    // 定义结构体用于存储趋势区间结果
    struct TrendInterval {
        double value;
        int start;
        int end;
        std::string trend;
    };

    // 中间区间结构体，用于合并处理
    struct Interval {
        int start;
        int end;
        double slope;
        Interval(int s, int e, double sl) : start(s), end(e), slope(sl) {}
    };
    void columnColsAverages(std::vector<std::vector<float>> &matrix_data, std::vector<float> &columnAverages);
    void columnRowAverages(std::vector<float> &matrix_data, float &columnAverages);
    double calculate_slope(const std::vector<double>& window);
    std::vector<Interval> merge_intervals(const std::vector<Interval>& intervals);
    void findTrendRegression(std::vector<float> &data, int window_size, float slope_threshold, std::vector<TrendInterval> &interval_data);
    void carpetFreespacePointToLinesaserFusion(cv::Mat &carpet_freespace_mask, EcoAInterfaceSlData_t& SLSData, CamerainnerInfo& paramCam, bool &carpet_mask_valid_flag,std::vector<std::vector<float>> &linelaser_to_cam,std::vector<float> &linelaser_ground_points_colmean,std::vector<std::vector<float>> &linelaser_z_grayscale_average);
    void secondVerificationCheckCarpetMaskVdlidFalg(cv::Mat &carpet_linelaser_map_probability, std::vector<cv::Point> &carpet_linelaser_valid_pose_vec, float pose_x, float pose_y, float theta, EcoInstanceObjectSeg *ecoinstanceobjectseg, bool &carpet_mask_valid_flag);


    //　获取　ｔｏｐｋ的分类结果  
    int rknn_cls_GetTop(int background, float *pfProb,  float *pfMaxProb,  int *pMaxClass,  int outputCount,  int topNum );


    // 人脸绑定到行人
    void face2person(EcoInstanceObjectSeg *ecotopcamoutputresult_);


    // 目标检测结果后处理，标签隐射+测距输出
    void topcamstataicdectect(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu);

    // 红外目标检测结果后处理，标签隐射+测距输出
    void topircamdectect(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu);

    // 分割结果后处理，标签隐射+测距输出
    void topcamstataicseg(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu, cv::Mat& single_rug_mask);

    // 分类结果后处理，标签隐射+测距输出
    void topcamstataiccls(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu);
    
    // 零拷贝 修改rknn_output为rknn_tensor_mem **，添加want_float传入
    // yolov7 家具检测
    void yolov7_post_process(EcoRknnModelParams& modelparams, rknn_output outputs[3],  std::vector<detect_result_t> &proposals,
                         int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_);
    
    // yolov8 家具检测
    void yolov8_indoor_post_process(EcoRknnModelParams& modelparams, rknn_output outputs[3],  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_);

   // yolov8 家具检测+方向+形状属性
    void yolov8_indoor_prop_post_process(EcoRknnModelParams& modelparams, rknn_output outputs[3],  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_);

    // yolov8 家具检测+方向+形状属性 零拷贝接口
    void yolov8_indoor_prop_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_, int want_float);

    void yolov8_indoor_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_, int want_float);

    void convert_image_with_letterbox(int model_in_w, int model_in_h, int image_in_w, int image_in_h, letterbox_t* letterbox);

    void Multitask_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                    int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch);

    // yolov8 多任务检测 零拷贝接口
    void multitask_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void freespace_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs,  cv::Mat &mask,  std::vector<EcoKeyPoint>& maskdata,
                        int image_in_w, int image_in_h, float clsthreshold_, float iouThreshold_, std::vector<std::vector<float>> out_threshold_of_all, int& cm_distance);

    // 地毯 零拷贝接口
    void freespace_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  cv::Mat &mask,  std::vector<EcoKeyPoint>& maskdata,
                            int image_in_w, int image_in_h, float clsthreshold_, float iouThreshold_, std::vector<std::vector<float>> out_threshold_of_all, int& cm_distance);

    void groundpoint_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs, std::vector<EcoKeyPoint>& maskdata,
                        int image_in_w, int image_in_h, float conf_threshold, int& cm_distance);

    // 接地点 零拷贝接口
    void groundpoint_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, std::vector<EcoKeyPoint>& maskdata,
                    int image_in_w, int image_in_h, float conf_threshold, int& cm_distance);

    void yolox_peopledet_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_);

    // 人形 零拷贝接口
    void yolox_peopledet_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                            int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_);

    int yolov8_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, 
                            std::vector<float> conf_threshold, std::vector<float> nms_threshold, float class_number, std::vector<detect_result_t>& od_results,
                            int start_idx, int start_label);

    // 多任务检测 零拷贝后处理
    int yolov8_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, letterbox_t *letter_box, 
                        std::vector<float> conf_threshold, std::vector<float> nms_threshold, float class_number, std::vector<detect_result_t>& proposals,
                        int want_float);

    int line_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, 
                        std::vector<float> conf_threshold, std::vector<float> line_box_threshold, float line_air_threshold_float, std::vector<detect_result_t>& od_results, 
                        int start_idx, int start_label);

    void yolov5_PM_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch);

    // 颗粒物污渍 零拷贝接口
    void yolov5_PM_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov5_Liquid_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov8_Liquid_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov5_lint_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov5_drystain_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);
        
    void yolov8_irstain_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov5_IR_detect_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch);

    void yolov5_animal_detect_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch);

    void yolov5_animal_detect_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov11_animal_detect_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void yolov8_obstacle_detect_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float);

    void dirt_det_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int modelSwitch);

    void wuzi_int8_det_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                            int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int modelSwitch);

    void bed_det_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int topk);

    // 分类 零拷贝接口
    void bed_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int topk);

    // 模型输出 int转float
    void int2float(int8_t *outputs, float *outputs_float, int size, int32_t zp, float scale);

    void groundpoint2rug(EcoInstanceObjectSeg* ecoCamOutputResult, EcoSegInference* ecoGroundPointsseg_, 
                        EcoSegInference* ecoFreespacetargetseg_, cv::Mat&  rug_mask);

    // 地毯点多帧匹配
    void rug_multiFrameMatch(std::vector<EcoKeyPoint>& Freespace_maskdata, EcoAInterfaceDeebotStatus_t pose, std::deque<std::unordered_set<int>>& coords_w_queue, int max_match_frame=3, int num_match_frame=2, int grid_size=25);

}

#endif //_POSTPROCESS_H_
