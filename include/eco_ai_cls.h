
/******************************************************************************
模块名　　　　：　eco_ai_cls
文件名　　　　：　eco_ai_cls.h
相关文件　　　：　eco_ai_cls.h
文件实现功能　：　分类识别(特征提取)头文件定义
作者　　　　　：　周峰
版本　　　　　：　1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        走读人    修改内容
2022/01/13    1.0                     周峰         创建
******************************************************************************/

#ifndef __ECO_AI_CLS_H__
#define __ECO_AI_CLS_H__

#include"inference.h"
#include"document.h"


#define MAX_EXTRO_BLOB_LEN 1024

#ifdef __cplusplus
extern "C" 
{
#endif

namespace sweeper_ai
{

    typedef enum EM_EcoAIClsTypes
    {
        EM_CLS_NONE = -1,       // 其他模型
        EM_CLS,                 // 分类模型
        EM_EXTRACTBLOBS,        // 特征提取模型
        EM_BOTH_CLS,            // 分类以及特征提取

        EM_OTHERCLSTYPES = 999

    }EcoAIClsTypes;


    class EcoObjectClsInference
    {
        public:
            EcoObjectClsInference();
            ~EcoObjectClsInference();

            EcoEStatus ecoObjectClsOpen(rapidjson::Value &param);
            EcoEStatus ecoObjectClsInfer(const std::vector<cv::Mat> &img,const cv::Rect &ROI_rect);
            EcoEStatus ecoObjectClsClose();

            EcoGroundObjectsCls* getObjectsCls()  { return& groundobjectscls_;} 
            EcoExtractBlobs*     getExtractBlobs(){ return& ecoextractblobs_;} 
            bool                 getOpenFlag()    { return bOpenFlag;}

            EcoEStatus showObjectCls(const cv::Mat& img) {return show_objects_cls(img);}

        private:

            bool bOpenFlag;                                     // 判断是否打开模型初始化句柄

            EcoAIClsTypes ecoaiclstypes_;                       // 分类还是特征提取类型

            int background_;                                    // 背景类
            int ntopkcls_;                                      // 分类ｔｏｐｋ个数
            int nextranum_;                                     // 特征提取ｂｌｏｂｓ个数
            float fclsThreshold_;                               // 输出的置信度阈值

            EcoResizeTypeS resize_type;

            EcoRknnModelParams modelparams;                     // rknn 模型参数

            cv::Mat  * resize_image_;                           // 输入网络的图片

            int8_t * input_data_;

            EcoGroundObjectsCls  groundobjectscls_;             //　目标的分类类别
            EcoExtractBlobs      ecoextractblobs_;              //　目标提取对比特征

            EcoEStatus show_objects_cls(const cv::Mat& img);
    };

}

#ifdef __cplusplus
}
#endif

#endif // __ECO_AI_DETECT_H__










