
/******************************************************************************
模块名      ： eco_ai_detect
文件名      ： eco_ai_detect.h
相关文件     ： eco_ai_detect.h
文件实现功能  ： 目标检测头文件定义
作者        ： 周峰
版本        ：　1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        走读人    修改内容
2022/01/08    1.0                     周峰         创建
******************************************************************************/

#ifndef __ECO_AI_DETECT_H__
#define __ECO_AI_DETECT_H__


#include"inference.h"
#include"document.h"


#ifdef __cplusplus
extern "C" 
{
#endif

namespace sweeper_ai
{
    typedef enum EM_EcoAIDetectTypes
    {
        EM_DECTECT_NONE = -1,
        EM_SSD,
        EM_YOLOV5_PM,
        EM_YOLOX_people_det,
        EM_YOLOV7_indoor_det,
        EM_YOLOV8_line_multitask_det,
        EM_DIRT_det,
        EM_MOBILENET_room_classify_det,
        EM_MOBILENET_bed_det,
        EM_YOLOV5_IR_OBJ,
        EM_YOLOV5_ANIMAL,
        EM_YOLOV8_indoor_prop,
        EM_YOLOV8_liquid,
        EM_YOLOV8_obstacle,
        EM_YOLOV5_lint,
        EM_YOLOV5_drystain,
        EM_YOLOV8_IR_STAIN,
        
        EM_OTHERDETECTTYPES = 999

    }EcoAIDetectTypes;


    class EcoDetectInference
    {
    public:
        EcoDetectInference();
        ~EcoDetectInference();

        EcoEStatus ecoDetectOpen(rapidjson::Value &param);              //　读取模型以及开辟各种所需的内存　
        EcoEStatus ecoDetectInfer(const std::vector<cv::Mat>  &img, const cv::Rect &ROI_rect, const int modelSwitch); //　对输入的图片进行处理，输出检测结果
        EcoEStatus ecoDetectClose();                                       //　释放申请的内存
        
        EcoGroundObjectDects * getDetectObjects() { return &ecogroundobjectdects_;} 
        bool                   getOpenFlag()      {return bOpenFlag;}
        std::vector<int>       getaftermodelids() { return after_model_ids;} 

        // EcoEStatus showDetectObjets(const cv::Mat& img) {return draw_objects(img);}
        EcoEStatus showDetectObjets(const cv::Mat& img, const EcoAInterfaceDeebotStatus_t& st, const std::string& image_path, cv::Mat& image) 
        {return draw_objects(img, st, image_path, image);}

        std::string ecoGetImagePath() {return image_save_path;}

    private:

        bool bOpenFlag;                                     // 判断是否打开模型初始化句柄

        std::vector<std::vector<float>>    out_threshold_of_each_;  //每个类的置信度阈值
        std::vector<std::vector<float>>    iouThreshold_of_each_;  //每个模型多个头的不同的IOU阈值
        std::vector<int>      after_model_ids;         //级联模型的编号
        int8_t ** poutputbuf;
        std::string           image_save_path;


        int   background_;                                  // 背景类标签
        int   ntopkcls_;                                    // 最大的目标检测类标个数
        int   nmaxdetectnum_;                               // 最大的目标检测输出个数 
        float detectThreshold_;                             // 输出的置信度阈值
        float iouThreshold_;                                // 过滤ｉｏｕ阈值

        EcoRknnModelParams modelparams;                     // rknn 模型参数

        EcoResizeTypeS resize_type;                         // 区分使用直接resize还是pad + resize
        int8_t  * input_data_;                              // CHW 输入
        cv::Mat * resize_image_;                            // 输入网络的图片

        EcoAIDetectTypes     detecttype_;                   // 目标检测算法类型
        EcoGroundObjectDects ecogroundobjectdects_;         // 目标检测输出结构体

        // EcoEStatus draw_objects(const cv::Mat& img);
        EcoEStatus draw_objects(const cv::Mat& img, const EcoAInterfaceDeebotStatus_t& st, std::string image_path, cv::Mat& image_before);
    };

}
#ifdef __cplusplus
}
#endif

#endif // __ECO_AI_DETECT_H__










