
/******************************************************************************
模块名　　　　： eco_task_infer
文件名　　　　： eco_task_infer.h
相关文件　　　： eco_task_infer.h
文件实现功能　： 业务任务头文件函数声明
作者　　　　　： 周峰
版本　　　　　： 1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        走读人    修改内容
2022/01/08    1.0                     周峰         创建
******************************************************************************/

#ifndef __ECO_TASK_INFER_H__
#define __ECO_TASK_INFER_H__

#include"eco_ai_detect.h"
#include"eco_ai_seg.h"
#include"eco_ai_cls.h"
#include"inference.h"

#ifdef __cplusplus
extern "C" {
#endif


namespace sweeper_ai
{

    class EcoTaskInference
    {
    public:

        EcoTaskInference();
        ~EcoTaskInference();
        //　参数设置（申请内存，打开模型）
        EcoEStatus ecoTaskOpen(char * config_str); 
        //　推断图片得到结果
        EcoEStatus ecoTaskInfer(const ImageDatas &input_data);        
        //　关闭模型，释放内存
        EcoEStatus ecoTaskClose();                         
        //　获取对应图片输出结果
        EcoInstanceObjectSeg* getCamDectInferResult(int model_ID)  {return  &ecocamoutputresult_[model_ID];}       //　根据　model_id 获取模型输出结果

        EcoEStatus showEcoTaskResult(const cv::Mat& img, bool bsave_image)    {return drawObjects(img, bsave_image);}

        EM_AI_TYPE getAIType(){return bflag;}

    private:

        // CameraInfo RGBDdown;                              //下镜头内外参数
        CameraInfo RGBDup;                                //上镜头内外参数
        // CameraInfo RGBDafter;                             //后镜头内外参数               

        EM_AI_TYPE bflag;  // 判断 红外AI 还是 RGB AI

        int     nirdetectmodelnum;
        EcoDetectInference*    ecoircamtargetdetect_;       //　多个目标检测功能类

        int     ndetectmodelnum;
        EcoDetectInference*    ecocamtargetdetect_;       //　多个目标检测功能类

        int     nsegmodelnum;
        EcoSegInference*       ecocamtargetseg_;          //　多个分割，点检测功能类

        int     nclsmodelnum;
        EcoObjectClsInference* ecocamtargetcls_;          //　多个分类识别功能类

        int     nimgnum; // nmodelnum
        EcoInstanceObjectSeg*  ecocamoutputresult_;       //　多个模型输出结果

        cv::Mat  rug_mask;                                //  单帧地毯 BEV 视角分割结果用于按距离过滤


        EcoEStatus drawObjects(const cv::Mat& img, bool& save_image); 
    };


}

#ifdef __cplusplus
}
#endif

#endif // __ECO_ASPEC_DETECT_H__


