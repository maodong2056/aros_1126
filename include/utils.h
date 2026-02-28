
#ifndef _UTILS_POSTPROCESS_H_
#define _UTILS_POSTPROCESS_H_


#include<unistd.h>
#include<sys/time.h>
#include"eco_task_infer.h"
#include "rga/im2d.h"
#include "rga/rga.h"
namespace sweeper_ai
{

    //　读取算法模型
    unsigned char *load_model(const char *filename, int *model_size);

    // rga+resize
    int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size);

    // resize+底部padding 输入图片
    void static_resize(cv::Mat& img, cv::Mat& resize_image, const int& INPUT_W , const int& INPUT_H) ;

    // resize+上下padding 输入图片
    void static_resize_top_bottom(cv::Mat& img, cv::Mat& resize_image, const int& INPUT_W , const int& INPUT_H) ;

    // resize 函数
    void eco_resize(cv::Mat& img, cv::Mat& resize_image, const int& INPUT_W , const int& INPUT_H, const EcoResizeTypeS& resize_type); 

    // rga_copy
    int rga_copy(rga_buffer_t &src, rga_buffer_t &dst, size_t img_width, size_t img_height, void *src_img, void *dst_img);

    //　字符分割,得到每个类的阈值
    void str_split(std::string &all_str, const std::string delimit, std::vector<float> &result);

    //　打印信息并且获取模型输入大小
    int rknn_get_ctx_attr(EcoRknnModelParams &modelparams);
    
    //　零拷贝 提前申请内存 打印信息并且获取模型输入大小
    int rknn_get_ctx_attr_zero_copy(EcoRknnModelParams &modelparams);
    
    //　读取镜头内外参  等参数
    EcoEStatus load_caminfo(int camid, std::string RGBD1_cam_yaml_path, CameraInfo& RGBDup);


    // 通道类型转换 CHW 转 HWC
    void nchw_to_nhwc(int8_t**src,  int8_t**det,  int weith_, int height_, int channel_);

    // 通道类型转换 HWC 转 CHW 
    void nhwc_to_nchw(int8_t** src, int8_t** det, int weith_, int height_, int channel_);

    void lds2pixel(std::vector< std::vector<float> >& lds_map, int64_t& image_time, EcoAInterfaceDeebotStatus_t& ImagePose, 
    EcoAInterfaceLdsData_t& LdsData, EcoAInterfaceSlData_t& SLSData, CamerainnerInfo& paramCam, std::vector<float>& Tcl);

    bool brug_seg(std::vector<EcoKeyPoint>& maskdata);

    bool brug_seg_remove(EcoKeyPoint& point_mask, cv::Mat& rug_mask, cv::Point3f& max_keypoint, cv::Point3f& min_keypoint, cv::Mat& single_rug_mask);

    // 地毯点和接地点融合仅在bev空间进行
    bool brug_seg_remove_bev(EcoKeyPoint& point_mask, cv::Mat& rug_mask, cv::Point3f& max_keypoint, cv::Point3f& min_keypoint, cv::Mat& single_rug_mask);

    std::time_t getTimeStamp(); 

    int checkPoseInPureTextureArea(EcoAInterfaceAreas_t* spotAreas, EcoAInterfaceDeebotStatus_t& pose);
}
#endif
