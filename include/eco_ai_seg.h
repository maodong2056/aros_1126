
/******************************************************************************
模块名　　　　： eco_ai_seg
文件名　　　　： eco_ai_seg.h
相关文件　　　： eco_ai_seg.h
文件实现功能　： 分割头文件定义
作者　　　　　： 周峰
版本　　　　　： 1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        创建人    修改内容
2022/01/08    1.0                     周峰         创建
******************************************************************************/

#ifndef __ECO_AI_SEG_H__
#define __ECO_AI_SEG_H__

#include"inference.h"
#include"document.h"
#include"string.h"


#ifdef __cplusplus
extern "C" {
#endif

namespace sweeper_ai
{

    typedef enum EM_EcoAISegTypes
    {
        EM_SEG_NONE = -1,
        EM_SEG,                    // 分割
        EM_BOTH_SEG,

        EM_POINTS_DETECT =10,      // 关键点检测
        EM_BOTH_POINTS_SEG,
        EM_freespace_seg,
        EM_OTHERSEGTYPES = 999

    }EM_EcoAISegTypes;


    class EcoSegInference
    {
    public:
        EcoSegInference();
        ~EcoSegInference();

        EcoEStatus ecoSegOpen(rapidjson::Value &param);              //　读取模型以及开辟各种所需的内存　
        EcoEStatus ecoSegInfer(const std::vector<cv::Mat> &img, const cv::Rect &ROI_rect, int& cm_distance); //　对输入的图片进行处理，输出检测结果
        EcoEStatus ecoSegClose();                                       //　释放申请的内存

        EcoInstanceObjectSeg* getSegMasks(){ return &ecoinstanceobjectseg;} 

        bool                  getOpenFlag(){ return bOpenFlag;}
        std::vector<int>      getaftermodelids() {return after_model_ids;} 


        EcoEStatus showSegMasks(const std::vector<EcoKeyPoint>& segmaskdata, const cv::Mat& img, const EcoAInterfaceDeebotStatus_t& st, std::string image_path) {return draw_objects(segmaskdata, img, st, image_path);}

        std::string ecoGetImagePath() {return image_save_path;}


        std::vector<float> linelaser_ground_points_colmean_; //结构光（地面的均值中心）
        std::vector<std::vector<float>> linelaser_z_grayscale_average;
        cv::Mat carpet_linelaser_map_probability = cv::Mat::zeros(1200, 1200, CV_8UC1);
        std::vector<cv::Point> carpet_linelaser_valid_pose_vec;


    private:

        bool bOpenFlag;                                     // 判断是否打开模型初始化句柄

        EM_EcoAISegTypes ecoaisegtypes_;                    // 模型类型

        std::vector<std::vector<float>> out_threshold_of_all;       //每个类的阈值

        std::vector<int>  after_model_ids;                  //级联模型的编号

        int8_t ** poutputbuf;
        std::string           image_save_path;
        
        int background_;                                    // 背景类信息
        float clsthreshold_;                                // 置信度阈值
        float iouThreshold_;                                // 过滤ｉｏｕ阈值

        // std::vector<float> linelaser_ground_points_colmean_; //结构光（地面的均值中心）
        // std::vector<std::vector<float>> linelaser_z_grayscale_average;


        EcoRknnModelParams modelparams;                 // rknn 模型参数

        cv::Mat  * resize_image_;                           // 输入网络的图片
        

        int8_t * input_data_;

        EcoResizeTypeS resize_type;                                    // 区分使用直接resize还是pad + resize

        EcoInstanceObjectSeg ecoinstanceobjectseg;          // 目标检测输出结构体

        // EcoEStatus draw_objects(const cv::Mat& img);
        EcoEStatus draw_objects(const std::vector<EcoKeyPoint>& segmaskdata, const cv::Mat& img, const EcoAInterfaceDeebotStatus_t& st, std::string image_path);
    };

}

#ifdef __cplusplus
}
#endif

#endif // __ECO_AI_DETECT_H__










