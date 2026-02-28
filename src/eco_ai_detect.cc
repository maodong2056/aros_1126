
/******************************************************************************
模块名　　　　： eco_ai_detect.h
文件名　　　　： eco_ai_detect.cc
相关文件　　　： eco_ai_detect.cc
文件实现功能　： 目标检测类源码
作者　　　　　： 周峰
版本　　　　　： 1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        创建人    修改内容
2022/01/08    1.0                     周峰         创建
******************************************************************************/

#include "eco_ai_detect.h"
#include "postprocess.h"
#include "utils.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int save_img_result_(1);
#define wuzitxt "/data/autostart/wuzi.txt" 
bool wuzi = false;

static std::vector<cv::Scalar> vcolor_inner={
    cv::Scalar(0, 0, 125)  , cv::Scalar(0, 0, 225)    , cv::Scalar(0, 125, 0)    , cv::Scalar(0, 125, 125)  , cv::Scalar(0, 255, 0)
    , cv::Scalar(0, 255, 125), cv::Scalar(0, 255, 255)  , cv::Scalar(125, 0, 0)    , cv::Scalar(125, 0, 125)  , cv::Scalar(125, 0, 255), cv::Scalar(125, 125, 0)
    , cv::Scalar(125, 255, 0), cv::Scalar(125, 125, 125), cv::Scalar(125, 125, 255), cv::Scalar(125, 255, 255), cv::Scalar(255, 0, 0)  , cv::Scalar(255, 125, 0)
    , cv::Scalar(255, 255, 0), cv::Scalar(255, 125, 125), cv::Scalar(255, 125, 255), cv::Scalar(255, 255, 255)};


namespace sweeper_ai
{

    EcoDetectInference::EcoDetectInference():
    bOpenFlag(false),   poutputbuf(NULL),   resize_image_(NULL), input_data_(NULL),
    nmaxdetectnum_(48), detecttype_(EM_DECTECT_NONE), detectThreshold_(0.24),   iouThreshold_(0.5), 
    ntopkcls_(1),       background_(-1),    ecogroundobjectdects_(EcoGroundObjectDects()), modelparams(EcoRknnModelParams()),
    resize_type(NONE_RESIZE)
    {
        out_threshold_of_each_.clear();
        iouThreshold_of_each_.clear();
        after_model_ids.clear();
    }

    EcoDetectInference::~EcoDetectInference()
    {

    }

    EcoEStatus EcoDetectInference::ecoDetectOpen(rapidjson::Value &param)
    { 
        int ret(0);
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型

        out_threshold_of_each_.clear();  //每个类的阈值
        iouThreshold_of_each_.clear();   //单个模型多个头中每个头对应的阈值

        after_model_ids.clear();
        poutputbuf = NULL;

        if (param.IsNull())
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "detectparam.IsNull() in EcoDetectInference::ecoDetectOpen"  << std::endl;
            return ecoestatus_;
        }

        nmaxdetectnum_ = MAX_DECTECT_NUM;                                    //　单帧最大的检测目标个数

        if (param.HasMember("model_type"))
        {
            detecttype_ = EcoAIDetectTypes(param["model_type"].GetInt());    //　目标检测模型算法类型
        }

        if (param.HasMember("background"))
        {
            background_ = param["background"].GetInt();                      //　背景类标签
        }

        if (param.HasMember("resize_type"))
        {
            resize_type = EcoResizeTypeS(param["resize_type"].GetInt());     //　图像预处理方式，resize（0） 还是 pad + resize（1）
        }

        if (param.HasMember("cls_topk"))
        {
            ntopkcls_ = param["cls_topk"].GetInt();                          //　目标检测topk分类
        }
 
        if (param.HasMember("detect_threshold"))
        {
            detectThreshold_ = param["detect_threshold"].GetFloat();         //　目标检测模型置信度阈值
        }

        if (param.HasMember("iou_threshold"))
        {
            rapidjson::Value IOU_threshold_of_each = param["iou_threshold"].GetArray();   
            for (rapidjson::SizeType j = 0; j < IOU_threshold_of_each.Size(); j++)
            {
                std::vector<float> IOU_tmpVec;
                for(rapidjson::SizeType i = 0; i < IOU_threshold_of_each[j].Size(); i++)
                {
                    IOU_tmpVec.push_back(IOU_threshold_of_each[j][i].GetFloat());
                }
                iouThreshold_of_each_.push_back(IOU_tmpVec);
            }  

        }

        if (param.HasMember("threshold"))
        {
            //　定制化每类阈值
            rapidjson::Value threshold_of_each = param["threshold"].GetArray();   
            for (rapidjson::SizeType j = 0; j < threshold_of_each.Size(); j++)
            {
                std::vector<float> Conf_tmpVec;
                for(rapidjson::SizeType i = 0; i < threshold_of_each[j].Size(); i++)
                {
                    Conf_tmpVec.push_back(threshold_of_each[j][i].GetFloat());
                }
                out_threshold_of_each_.push_back(Conf_tmpVec);
            }  
        }

        ////从这里读取配置文件中"after_model",看其中是否跟着级联模型
        if (param.HasMember("after_model"))
        {
            //　定制化每类阈值
            rapidjson::Value after_models = param["after_model"].GetArray();   
            for (rapidjson::SizeType j = 0; j < after_models.Size(); j++)
            {
                after_model_ids.push_back(after_models[j].GetFloat());
            }  
        }

        if (param.HasMember("img_save"))
        {
            image_save_path = param["img_save"].GetString();    //　目标检测模型结构文件
            if(!image_save_path.empty())
            {
                if(-1 == access(image_save_path.c_str(), 0))
                {
                    // std::cout << "image_save_path = " << image_save_path << std::endl;
                    // 创建文件夹
                    int result = mkdir(image_save_path.c_str(), 0755);
                
                    if (result == 0) 
                    {
                        std::cout << std::endl << std::endl;
                        std::cout << "Folder created successfully." << std::endl;
                    } 
                    else 
                    {
                        std::cout << std::endl << std::endl;
                        std::cout << "Failed to create folder." << std::endl;
                    }
                }
            }
        }

        if (param.HasMember("name"))
        {
            // 初始化input_mems和output_mems数组中的所有元素为NULL
            modelparams.input_mems[0] = NULL;
            for (int i = 0; i < 15; i++) {
                modelparams.output_mems[i] = NULL;
            }

            int model_data_size = 0; // 模型大小

            std::string model_name_ = param["name"].GetString();    //　目标检测模型结构文件
            std::cout << "model_name_ = " << model_name_ << std::endl;
            //判断模型文件是否存在
            if(-1 == access(model_name_.c_str(), 0))
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "model_file is empty, model_name_ = " << model_name_ << std::endl;
                ecoDetectClose();
                return ecoestatus_;
            }

            modelparams.model_data = load_model(model_name_.c_str(), &model_data_size);
            if (NULL == modelparams.model_data)
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "load_model error ret = " << ret << std::endl;
                ecoDetectClose();
                return ecoestatus_;
            }

            ret = rknn_init(&modelparams.ctx, modelparams.model_data, model_data_size, 0, NULL);    //　ctx　环境句柄初始化
            if (ret < 0)
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "rknn_init error ret = " << ret << std::endl;
                ecoDetectClose();
                return ecoestatus_;
            }
            if (NULL != modelparams.model_data)
            {
                free(modelparams.model_data);
                modelparams.model_data = NULL;
            }
        }
        else
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "have no model file"<< std::endl;
            ecoDetectClose();
            return ecoestatus_;
        }

        //　获取设定的npu_id并设置模型所在npu
        rknn_core_mask core_mask = rknn_core_mask(param["core"].GetInt());
        if (core_mask < RKNN_NPU_CORE_AUTO || core_mask > RKNN_NPU_CORE_0_1_2)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "NPU_ID is ERROR npu_id = " << param["core"].GetInt() << std::endl;
            ecoDetectClose();
            return ecoestatus_;
        }

        // 设置运行　ＩＤ--在rk3562中不需要--打开会报错
        // ret = rknn_set_core_mask(modelparams.ctx, core_mask);
        // if (ret < 0)
        // {
        //     ecoestatus_ = EStatus_InvalidParameter;
        //     std::cout << "rknn_set_core_mask error ret = " << ret << std::endl;
        //     ecoDetectClose();
        //     return ecoestatus_;
        // }

        //　获取模型输入和输出节点个数                    
        ret = rknn_query(modelparams.ctx, RKNN_QUERY_IN_OUT_NUM, &modelparams.io_num, sizeof(modelparams.io_num));
        if (ret < 0)
        {
            printf("rknn_query rknn_input_output_num error ret = %d\n", ret);
            ecoDetectClose();
            return ecoestatus_;
        }

        poutputbuf         = new int8_t* [3];  
        modelparams.nmodelinputchannel_ = new int[modelparams.io_num.n_input];               // 网络输入　通道数
        modelparams.nmodelinputweith_   = new int[modelparams.io_num.n_input];               // 网络输入　宽度
        modelparams.nmodelinputheight_  = new int[modelparams.io_num.n_input];               // 网络输入　高度
        resize_image_      = new cv::Mat[modelparams.io_num.n_input];                        // 网络输入　数据（预处理之后）

        modelparams.nmodeloutputchannel_ = new int[modelparams.io_num.n_output];             // 网络输出　通道数
        modelparams.nmodeloutputweith_   = new int[modelparams.io_num.n_output];             // 网络输出　宽度
        modelparams.nmodeloutputheight_  = new int[modelparams.io_num.n_output];             // 网络输出　高度

        input_data_          = new int8_t[MAX_MALLOC_NUM];                         // 输出节点转换内存

        if (NULL == poutputbuf  || NULL == modelparams.nmodelinputchannel_   
         || NULL == modelparams.nmodelinputweith_   || NULL == modelparams.nmodelinputheight_ 
         || NULL == resize_image_ || NULL == input_data_ || NULL == modelparams.nmodeloutputchannel_ 
         || NULL == modelparams.nmodeloutputweith_ || NULL == modelparams.nmodeloutputheight_)
        {
            ecoestatus_ = EStatus_OutOfMemory;
            std::cout  <<  "NULL == nimgresizechannel_ || NULL == nimgresizeweith_ || NULL == nimgresizeheight_ || NULL == resize_image_ || NULL == input_data_ can't memory, ecoestatus_ = " <<  ecoestatus_ <<std::endl;
            ecoDetectClose();
            return ecoestatus_;
        }

        for (size_t outnum = 0; outnum < 3; outnum++)
        {
            poutputbuf[outnum] = NULL;
        }

        //　通过句柄获取　输入节点参数
        // 零拷贝修改rknn_get_ctx_attr，添加输入输出的内存申请
        // ret = rknn_get_ctx_attr(modelparams);
        std::cout << "----------use zero copy---------" << std::endl; 
        ret = rknn_get_ctx_attr_zero_copy(modelparams);
        if (ret < 0)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_get_ctx_attr error ret = " << ret << std::endl;
            ecoDetectClose();
            return ecoestatus_;
        }

        for (size_t imginput = 0; imginput < modelparams.io_num.n_input; imginput++)
        {
            resize_image_[imginput] = cv::Mat(modelparams.nmodelinputheight_[imginput], modelparams.nmodelinputweith_[imginput],  CV_8UC3, cv::Scalar(114, 114, 114));
        }

        //　重置输出初始化
        if (NULL != ecogroundobjectdects_.ecogroundobject)
        {
            ecoDetectClose();
        }
        
        //　初始化输出结构体，根据最大的输出个数申请结果输出内存
        ecogroundobjectdects_.ecogroundobject = new EcoGroundObjectDect[nmaxdetectnum_];
        if (NULL == ecogroundobjectdects_.ecogroundobject)
        {
            ecoestatus_ = EStatus_OutOfMemory;
            std::cout  <<  "ecogroundobjectdects_.ecogroundobject can't memory, ecoestatus_ = " <<  ecoestatus_ <<std::endl;
            ecoDetectClose();
            return ecoestatus_;
        }

        //模型 ID 编号
        if (param.HasMember("id"))
        {
            ecogroundobjectdects_.model_id = param["id"].GetInt();                //　分类以及特征提取定义
        }

        //　申请目标检测所需的内存大小
        for (int nobj = 0; nobj < nmaxdetectnum_; nobj++)
        {
            ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.nobjectsclsnum         = ntopkcls_;
            ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.ptrecogroundobjectscls = new EcoGroundObjectCls[ntopkcls_];
            if (NULL == ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.ptrecogroundobjectscls)
            {
                ecoestatus_ = EStatus_OutOfMemory;
                std::cout  <<  "ecogroundobjectdects_.ecogroundobject[ " << nobj << " ].groundobjectsCls.ptrecogroundobjectscls can't memory, ecoestatus_ = " <<  ecoestatus_ << std::endl;
                ecoDetectClose();
                return ecoestatus_;
            }
        }

        bOpenFlag =true;

        if (access(wuzitxt, 0) == 0) 
        {
            wuzi = true;
        }

        return EStatus_Success;
    }


    EcoEStatus EcoDetectInference::ecoDetectInfer(const std::vector<cv::Mat> &img, const cv::Rect &ROI_rect, const int modelSwitch)
    {
        int ret(0);
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型

        if (img.empty())
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout  <<  "img.empty() in EcoDetectInference::ecoDetectInfer" <<  ecoestatus_ << std::endl;
            return ecoestatus_;
        }

        if ( img.size() < modelparams.io_num.n_input)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout  <<  "img.size():" << img.size() << " < io_num.n_input:" << modelparams.io_num.n_input << " in EcoDetectInference::ecoDetectInfer" <<  ecoestatus_ << std::endl;
            return ecoestatus_;
        }

        // 零拷贝省略rknn_input
        for (size_t i = 0; i < modelparams.io_num.n_input; i++)
        {

            if (img[i].empty())
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout  <<  "input img[ " << i << " ] is empty in EcoDetectInference::ecoDetectInfer" <<  ecoestatus_ << std::endl;
                return ecoestatus_;
            }
            ////uchar*,这里的bgr和img[i]的data的数据地址相同
            cv::Mat bgr = img[i](ROI_rect);

            if(bgr.cols != modelparams.nmodelinputweith_[i] || bgr.rows != modelparams.nmodelinputheight_[i]) 
            {
                // 零拷贝修改resize_image地址为预先申请内存的地址
                resize_image_[i].data = (unsigned char*)modelparams.input_mems[0]->virt_addr;
                eco_resize(bgr, resize_image_[i], modelparams.nmodelinputweith_[i], modelparams.nmodelinputheight_[i], resize_type);
            }
            else
            {
                // 零拷贝 rga加速拷贝到预先申请内存的地址
                rga_buffer_t src;
                rga_buffer_t dst;
                memset(&src, 0, sizeof(src));
                memset(&dst, 0, sizeof(dst));
                int ret = rga_copy(src, dst, modelparams.nmodelinputweith_[i], modelparams.nmodelinputheight_[i], (void *)bgr.data, (void *)modelparams.input_mems[i]->virt_addr);
                if (ret != IM_STATUS_SUCCESS)
                {
                    std::cout << "rga copy error!!!  " << ret << std::endl;
                }
            }
        }
        
        // 零拷贝省略rknn_output和rknn_inputs_set，仅保留want_float
        int want_float = -1;
        if(EM_YOLOV7_indoor_det         == detecttype_ 
        || EM_YOLOV8_line_multitask_det == detecttype_ 
        || EM_YOLOV5_PM                 == detecttype_ 
        || EM_YOLOV5_IR_OBJ             == detecttype_
        || EM_YOLOV5_ANIMAL             == detecttype_
        || EM_YOLOV8_indoor_prop        == detecttype_
        || EM_YOLOV8_liquid             == detecttype_
        || EM_YOLOV8_obstacle           == detecttype_
        || EM_YOLOV5_lint               == detecttype_
        || EM_YOLOV5_drystain           == detecttype_
        || EM_YOLOV8_IR_STAIN           == detecttype_)
        {   
            want_float = 0;
        }
        else if(EM_YOLOX_people_det            == detecttype_ 
             || EM_DIRT_det                    == detecttype_ 
             || EM_MOBILENET_room_classify_det == detecttype_  
             || EM_MOBILENET_bed_det           == detecttype_)
        {   
            want_float = 1;
        }

    auto end2 = std::chrono::system_clock::now();

        // Run
        ret = rknn_run(modelparams.ctx, nullptr);
        if(ret < 0) 
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_run fail! ret = " << ret << std::endl;
            return ecoestatus_;
        }

    auto end4 = std::chrono::system_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end2);
    std::cout << "kkkkkkkkkkkkkkkyolox_post_process time =" << duration4.count() << "ms" << std::endl;


        #ifdef D_DEBUG
            //　获取当前时间并计算运行时间
            rknn_perf_run perf_run;
            ret = rknn_query(modelparams.ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
            std::cout << "rknn_run time =" << perf_run.run_duration/1000.0 << "ms" << std::endl; 
        #endif


#ifdef D_DEBUG
    auto end2 = std::chrono::system_clock::now();
#endif
        std::vector<detect_result_t> proposals;

/******** 家具识别 ***********************************************************************************************/
        if (EM_YOLOV7_indoor_det == detecttype_)
        {
            std::cout << "start EM_YOLOV8_indoor_det " << std::endl;
            proposals.clear();

            int image_in_w = 1280;           ////强制改为1280*960
            int image_in_h = 960;         
            // yolov7_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_);
            // yolov8_indoor_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_);
            std::cout << "stop EM_YOLOV8_indoor_det, num of detect " ;
        }
/******** 家具识别 + 方向属性  ***********************************************************************************************/
        else if (EM_YOLOV8_indoor_prop == detecttype_)
        {
            std::cout << "start EM_YOLOV8_indoor_prop " << std::endl;
            proposals.clear();

            int image_in_w = 1280;           ////强制改为1280*960
            int image_in_h = 960;         
            // yolov7_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_);
            // yolov8_indoor_prop_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_);
            // 零拷贝
            // yolov8_indoor_prop_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, want_float);
            yolov8_indoor_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, want_float);
            std::cout << "stop EM_YOLOV8_indoor_prop, num of detect " ;
        }
/******** 多任务障碍物 ***********************************************************************************************/
        else if (EM_YOLOV8_line_multitask_det == detecttype_)
        {
            std::cout << "start EM_YOLOV8_line_multitask_det " << std::endl;
            proposals.clear();

            int image_in_w = 1280;           ////强制改为1280*960
            int image_in_h = 960;             
            // Multitask_postprocess(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch);
            // 零拷贝 仅保留检测头的后处理
            multitask_postprocess_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            std::cout << "stop EM_YOLOV8_line_multitask_det, num of detect " ;
        }
/******** 人形检测  ***********************************************************************************************/
        else if(EM_YOLOX_people_det == detecttype_)
        {
            std::cout << "start EM_YOLOX_people_det " << std::endl;
            proposals.clear();

            int image_in_w = img[0].cols;    // 这里得到的image_in_w是1440
            int image_in_h = img[0].rows;    // 这里得到的image_in_h是1080
            // yolox_peopledet_postprocess(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_);
            // 零拷贝 添加int8转float32
            yolox_peopledet_postprocess_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_);
            std::cout << "stop EM_YOLOX_people_det, num of detect " ;
        }
/******** 颗粒物 + 污渍识别  ***********************************************************************************************/
        else if(EM_YOLOV5_PM == detecttype_)
        {
            if (access(wuzitxt, 0) == 0) 
            {
                wuzi = true;
            }
            else
            {
                wuzi = false;
            }
            if(modelSwitch == 2 || modelSwitch == 3 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
            {
                std::cout << "start EM_YOLOV_PM " << std::endl;
            }
            // if(modelSwitch == 4 || modelSwitch == 5 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
            // {
                // std::cout << "start EM_YOLOV_DIRT_det " << std::endl;
            // }

            proposals.clear();
            std::vector<detect_result_t> proposals_tmp; 
            proposals_tmp.clear();

            int image_in_w = 512;           // 强制改为1280*960
            int image_in_h = 384;         
            // yolov5_PM_post_process(modelparams, outputs, proposals_tmp, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch);      
            // 零拷贝
            yolov5_PM_post_process_zero_copy(modelparams, modelparams.output_mems, proposals_tmp, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            for(int k = 0; k < proposals_tmp.size(); k++)
            {
                if(proposals_tmp[k].box.top <= 523 || proposals_tmp[k].box.top >= 955 || proposals_tmp[k].prop[0].condidence < 0.7
                || abs(proposals_tmp[k].box.left - proposals_tmp[k].box.right) > 1150)
                {
                    continue;
                }

                if(proposals_tmp[k].prop[0].name == 0)
                {
                    if(modelSwitch == 2 || modelSwitch == 3 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
                    {
                        proposals.push_back(proposals_tmp[k]);
                    }
                }
                // else if(proposals_tmp[k].prop[0].name != 0)
                // {
                //     // if(modelSwitch == 4 || modelSwitch == 5 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
                //     // {

                //         BOX_RECT& box = proposals_tmp[k].box;
                //         int w         = box.right - box.left;
                //         int h         = box.bottom - box.top;
                //         // 地面颜色均值
                //         int ground = img[0].at<cv::Vec3b>(box.top / 2.5, box.left / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.top / 2.5, box.left / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.top / 2.5, box.left  / 2.5)[2] * 0.11
                //         + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.right / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.right / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.right / 2.5)[2] * 0.11
                //         + img[0].at<cv::Vec3b>(box.top    / 2.5, box.right / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.top    / 2.5, box.right / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.top    / 2.5, box.right / 2.5)[2] * 0.11
                //         + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.left  / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.left  / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.left  / 2.5)[2] * 0.11; //读取像素
                //         // 污渍颜色均值
                //         int wzi = img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[2] * 0.11
                //         + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[2] * 0.11
                //         + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[2] * 0.11
                //         + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[2] * 0.11;

                //         // std::cout << "ground = " << ground / 4<< "     " << "wuzi = " << wzi / 4<< std::endl; 
                //         if(ground / 4 > 90 && wzi / 4 > 100 && proposals_tmp[k].prop[0].name == 2)
                //         {
                //             proposals_tmp[k].prop[0].name = 6;  // 区分出浅色地面上的浅色污渍
                //         }
                //         proposals.push_back(proposals_tmp[k]);

                //     // }
                // }
            }

            if(modelSwitch == 2 || modelSwitch == 3 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
            {
                std::cout << "stop EM_YOLOV_PM, num of detect " ;
            }
            // if(modelSwitch == 4 || modelSwitch == 5 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
            // {
                //  std::cout << "stop EM_YOLOV_DIRT_det, num of detect " ;
            // }         
        }
/******** 水渍 + 污渍识别  ***********************************************************************************************/
        else if(EM_YOLOV8_liquid == detecttype_)
        {
            // if(modelSwitch == 4 || modelSwitch == 5 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
            // {
                std::cout << "start EM_YOLOV8_liquid " << std::endl;
            // }

            proposals.clear();
            std::vector<detect_result_t> proposals_tmp; 
            proposals_tmp.clear();

            int image_in_w = 1280;           // 强制改为1280*960
            int image_in_h = 960;         
            yolov8_Liquid_post_process_zero_copy(modelparams, modelparams.output_mems, proposals_tmp, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            // yolov5_Liquid_post_process_zero_copy(modelparams, modelparams.output_mems, proposals_tmp, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            for(int k = 0; k < proposals_tmp.size(); k++)
            {
                if(proposals_tmp[k].box.top <= 523 || proposals_tmp[k].box.bottom >= 955 || proposals_tmp[k].prop[0].condidence < 0.86
                || abs(proposals_tmp[k].box.left - proposals_tmp[k].box.right) > 1150)
                {
                    continue;
                }

                // if(proposals_tmp[k].prop[0].name == 1)
                // {
                //     // SHA场景新增过滤条件，去掉混合态
                //     continue;
                // }
                if(proposals_tmp[k].prop[0].name == 4)
                {
                    proposals.push_back(proposals_tmp[k]);
                }

                if(proposals_tmp[k].prop[0].name != 0 && proposals_tmp[k].prop[0].name != 1 && proposals_tmp[k].prop[0].name != 4)
                {
                    // if(modelSwitch == 4 || modelSwitch == 5 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
                    // {

                        BOX_RECT& box = proposals_tmp[k].box;
                        int w         = box.right - box.left;
                        int h         = box.bottom - box.top;
                        // 地面颜色均值
                        int ground = img[0].at<cv::Vec3b>(box.top / 2.5, box.left / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.top / 2.5, box.left / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.top / 2.5, box.left  / 2.5)[2] * 0.11
                        + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.right / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.right / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.right / 2.5)[2] * 0.11
                        + img[0].at<cv::Vec3b>(box.top    / 2.5, box.right / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.top    / 2.5, box.right / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.top    / 2.5, box.right / 2.5)[2] * 0.11
                        + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.left  / 2.5)[0] * 0.3 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.left  / 2.5)[1] * 0.59 + img[0].at<cv::Vec3b>(box.bottom / 2.5, box.left  / 2.5)[2] * 0.11; //读取像素
                        // 污渍颜色均值
                        int wzi = img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[2] * 0.11
                        + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 7 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[2] * 0.11
                        + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 7 * w / 15)/2.5)[2] * 0.11
                        + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[0] * 0.3 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[1] * 0.59 + img[0].at<cv::Vec3b>((box.top + 8 * h / 15)/2.5, (box.left + 8 * w / 15)/2.5)[2] * 0.11;

                        // std::cout << "ground = " << ground / 4<< "     " << "wuzi = " << wzi / 4<< std::endl; 
                        if(ground / 4 > 90 && wzi / 4 > 90 && (proposals_tmp[k].prop[0].name == 2 || proposals_tmp[k].prop[0].name == 3))
                        {
                            proposals_tmp[k].prop[0].name = 50;  // 区分出浅色地面上的浅色污渍
                        }
                        proposals.push_back(proposals_tmp[k]);

                    // }
                }
            }
            // if(modelSwitch == 4 || modelSwitch == 5 || modelSwitch == 6 || modelSwitch == 7 || wuzi)
            // {
                std::cout << "stop EM_YOLOV8_liquid, num of detect " ;
            // }    
        }
/******** 污渍识别  ***********************************************************************************************/
        else if(EM_DIRT_det == detecttype_)
        {
            std::cout << "start EM_DIRT_det " << std::endl;
            proposals.clear();

            int image_in_w = 1280;          // 强制改为1280*960
            int image_in_h = 440;         
            // dirt_det_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, modelSwitch);
            std::cout << "stop EM_DIRT_det, num of detect " ;
        }
/******** 低矮区域识别  ***********************************************************************************************/
        else if(EM_MOBILENET_bed_det == detecttype_)
        {
            std::cout << "start EM_MOBILENET_bed_det " << std::endl;
            proposals.clear();

            int image_in_w = 1280;           ////强制改为1280*960
            int image_in_h = 960;         
            ////目前不设置阈值，保留接口
            // bed_det_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, ntopkcls_);
            // 零拷贝
            bed_det_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, ntopkcls_);
            std::cout << "stop EM_MOBILENET_bed_det, proposals[0] = " << proposals[0].prop[0].name;
        }
/******** 地面材质识别  ***********************************************************************************************/
        else if(EM_MOBILENET_room_classify_det == detecttype_)
        {
            std::cout << "start EM_MOBILENET_room_classify_det " << std::endl;
            proposals.clear();
            
            int image_in_w = 1280;           ////强制改为1280*960
            int image_in_h = 960;         
            ////目前不设置阈值，保留接口---
            // bed_det_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, ntopkcls_);
            // 零拷贝
            bed_det_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, ntopkcls_);
            std::cout << "stop EM_MOBILENET_room_classify_det, proposals[0] = " << proposals[0].prop[0].name;
        }
/******** 红外图识别  ***********************************************************************************************/
        else if(EM_YOLOV8_IR_STAIN == detecttype_)
        {
            std::cout << "start EM_YOLOV8_IR_STAIN " << std::endl;
            proposals.clear();

            int image_in_w = img[0].cols;
            int image_in_h = img[0].rows;       
            yolov8_irstain_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);      
            for(int k = 0; k < proposals.size(); k++)
            {
                int threshold_idx = proposals[k].prop[0].name;
                if (threshold_idx >= out_threshold_of_each_[0].size()) {
                    threshold_idx = 0;
                }
                if(proposals[k].prop[0].condidence < out_threshold_of_each_[0][threshold_idx])
                {
                    proposals[k].prop[0].name = 50;
                }
            }
            std::cout << "stop EM_YOLOV8_IR_STAIN, num of detect " ;
        }
/******** 宠物识别  ***********************************************************************************************/
        else if(EM_YOLOV5_ANIMAL == detecttype_)
        {
            std::cout << "start EM_YOLOV5_ANIMAL " << std::endl;
            proposals.clear();
            std::vector<detect_result_t> proposals_tmp; 
            proposals_tmp.clear();

            int image_in_w = 1280;           // 强制改为1280*960
            int image_in_h = 960;         
            // yolov5_animal_detect_post_process(modelparams, outputs, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch);      
            // yolov5_animal_detect_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            yolov11_animal_detect_post_process_zero_copy(modelparams, modelparams.output_mems, proposals_tmp, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            
            std::vector<detect_result_t> face_boxes;
            std::vector<detect_result_t> body_boxes;

            for (int k = 0; k < proposals_tmp.size(); k++) {
                if (proposals_tmp[k].prop[0].name == 0) {
                    body_boxes.push_back(proposals_tmp[k]);
                } else if (proposals_tmp[k].prop[0].name == 1) {
                    face_boxes.push_back(proposals_tmp[k]);
                }
            }

            for (int i = 0; i < body_boxes.size(); i++) {
                bool has_overlap_with_face = false;
                for (int j = 0; j < face_boxes.size(); j++) {
                    bool no_overlap = (body_boxes[i].box.right < face_boxes[j].box.left) || 
                                    (body_boxes[i].box.left > face_boxes[j].box.right) || 
                                    (body_boxes[i].box.bottom < face_boxes[j].box.top) || 
                                    (body_boxes[i].box.top > face_boxes[j].box.bottom);
            
                    if (!no_overlap) {
                        has_overlap_with_face = true;
                        break;
                    }
                }

                if(body_boxes[i].box.left < 10 || body_boxes[i].box.right > 1270){
                    body_boxes[i].prop[0].name = 50;
                }
                else{
                    if (has_overlap_with_face) {
                        body_boxes[i].prop[0].name = 0;
                    } else {
                        body_boxes[i].prop[0].name = 50;
                    }
                }
                proposals.push_back(body_boxes[i]);
            }

            std::cout << "stop EM_YOLOV5_ANIMAL, num of detect " ;
        }
/******** 障碍物识别  ***********************************************************************************************/
        else if(EM_YOLOV8_obstacle == detecttype_)
        {
            std::cout << "start EM_YOLOV8_obstacle " << std::endl;
            proposals.clear();

            int image_in_w = 1280;           // 强制改为1280*960
            int image_in_h = 960;         
            yolov8_obstacle_detect_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            std::cout << "stop EM_YOLOV8_obstacle, num of detect " ;
        }
/******** 毛絮识别  ***********************************************************************************************/
        else if(EM_YOLOV5_lint == detecttype_)
        {
            std::cout << "start EM_YOLOV5_lint " << std::endl;
            proposals.clear();

            int image_in_w = 512;           // 强制改为1280*960
            int image_in_h = 384;         
            yolov5_lint_post_process_zero_copy(modelparams, modelparams.output_mems, proposals, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            std::cout << "stop EM_YOLOV5_lint, num of detect " ;
        }
/******** 干涸污渍识别  ***********************************************************************************************/
        else if(EM_YOLOV5_drystain == detecttype_)
        {
            std::cout << "start EM_YOLOV5_drystain " << std::endl;
            proposals.clear();
            std::vector<detect_result_t> proposals_tmp; 
            proposals_tmp.clear();

            int image_in_w = 512;           // 强制改为1280*960
            int image_in_h = 384;         
            yolov5_drystain_post_process_zero_copy(modelparams, modelparams.output_mems, proposals_tmp, image_in_w, image_in_h, detectThreshold_, out_threshold_of_each_, iouThreshold_of_each_, modelSwitch, want_float);
            if (modelSwitch == 1)
            {
                std::cout << "EM_YOLOV5_drystain in pure ground" << std::endl;
                for(int k = 0; k < proposals_tmp.size(); k++)
                {
                    if(proposals_tmp[k].box.top <= 523 || proposals_tmp[k].box.bottom >= 955
                    || abs(proposals_tmp[k].box.left - proposals_tmp[k].box.right) > 1150)
                    {
                        continue;
                    }
                    proposals.push_back(proposals_tmp[k]);
                }
            }
            for(int k = 0; k < proposals_tmp.size(); k++)
            {
                if(proposals_tmp[k].box.top <= 523 || proposals_tmp[k].box.bottom >= 955
                || abs(proposals_tmp[k].box.left - proposals_tmp[k].box.right) > 1150)
                {
                    continue;
                }
                if(proposals_tmp[k].prop[0].name == 3)
                {
                    proposals.push_back(proposals_tmp[k]);
                }
            }
            std::cout << "stop EM_YOLOV5_drystain, num of detect " ;
        }
        ////这里得到最终的结果存储在proposals中，直接读取出来画图即可
        int k = 0;
        for (int i = 0; i < proposals.size(); i++)
        {
            // 内部第　ｉ　个输出结果，被删除，则不再取用
            if (!proposals[i].issure)
            {
                continue;
            }

            if (k >= nmaxdetectnum_)
            {
                break;
            }

            // 将内部第 i 个目标转成对外第 k 个输出结果                
            detect_result_t *det_result = &(proposals[i]);

            EcoGroundObjectDect *out_det_result = &(ecogroundobjectdects_.ecogroundobject[k]);
            out_det_result->position.clear();
            out_det_result->lds_position.clear();
            out_det_result->bisobjects     = false;
            out_det_result->bisface        = false;
            out_det_result->bisheypoint    = false;
            out_det_result->bisextractblob = false;
            out_det_result->groundobjectsCls.ptrecogroundobjectscls[0].fconfidence = -1;
            out_det_result->bisobjects  = true;


            // if(EM_YOLOV5_PM == detecttype_ || EM_DIRT_det == detecttype_
            // || EM_MOBILENET_room_classify_det == detecttype_ || EM_MOBILENET_bed_det == detecttype_
            // || EM_YOLOV8_liquid == detecttype_)
            // {
                out_det_result->rect.x      = MAX(1, int(det_result->box.left));
                out_det_result->rect.y      = MAX(1, int(det_result->box.top ));
            // }
            // else
            // {
            //     out_det_result->rect.x      = MAX(1, int(det_result->box.left) + ROI_rect.x);
            //     out_det_result->rect.y      = MAX(1, int(det_result->box.top ) + ROI_rect.y);
            // }
            ////经过后处理输出来就是处理好的框的坐标
            if(EM_YOLOX_people_det == detecttype_)
            {
                out_det_result->rect.width  = MIN(img[0].cols - out_det_result->rect.x - 1, int((det_result->box.right - det_result->box.left)));
                out_det_result->rect.height = MIN(img[0].rows - out_det_result->rect.y - 1, int((det_result->box.bottom - det_result->box.top)));        
            }
            else
            {
                out_det_result->rect.width  = MIN(1280 - out_det_result->rect.x - 1, int((det_result->box.right - det_result->box.left)));
                out_det_result->rect.height = MIN(960  - out_det_result->rect.y - 1, int((det_result->box.bottom - det_result->box.top)));
            }

#ifdef D_DEBUG
            std::cout<< "label =" << det_result->prop[0].name << " prop = " << det_result->prop[0].condidence << " x = " <<out_det_result->rect.x << " y = " << out_det_result->rect.y << " x2 = " << out_det_result->rect.x + out_det_result->rect.width << 
                " y2 = " << out_det_result->rect.y +  out_det_result->rect.height << std::endl;
#endif

            for (int ncls = 0; ncls < MIN(det_result->prop.size(), ntopkcls_); ncls++)
            {
                /////这里的inlabel是一个检测框中可能存在多个类别，该name就是根据模型得到的cls_id(txt中对应类别的索引)直接赋予得到的，这里再把name直接赋予inlabel
                out_det_result->groundobjectsCls.ptrecogroundobjectscls[ncls].inlabel     = det_result->prop[ncls].name;
                out_det_result->groundobjectsCls.ptrecogroundobjectscls[ncls].fconfidence = det_result->prop[ncls].condidence;
            }
            if(det_result->sub_prop.size() > 0)
            {
                out_det_result->objectprop.inlabel = det_result->sub_prop[0].name;
            }
            k++;         
        } 
        std::cout << " k = " << k << std::endl;
        ecogroundobjectdects_.ngroundobjectnum = k;

#ifdef D_DEBUG
    auto end4 = std::chrono::system_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end2);
    std::cout << "yolox_post_process time =" << duration4.count() << "ms" << std::endl;
#endif

        return EStatus_Success;
    }


    EcoEStatus EcoDetectInference::ecoDetectClose()
    {
        if (NULL != poutputbuf)
        {
            delete[] poutputbuf;
            poutputbuf = NULL;
        }

        if (NULL != modelparams.nmodelinputchannel_)
        {
            delete[] modelparams.nmodelinputchannel_;
            modelparams.nmodelinputchannel_ = NULL;
        }

        if (NULL != modelparams.nmodelinputweith_)
        {
            delete[] modelparams.nmodelinputweith_;
            modelparams.nmodelinputweith_ = NULL;
        }

        if (NULL != modelparams.nmodelinputheight_)
        {
            delete[] modelparams.nmodelinputheight_;
            modelparams.nmodelinputheight_ = NULL;
        }

        // 释放输出通道数
        if (NULL != modelparams.nmodeloutputchannel_)
        {
            delete[] modelparams.nmodeloutputchannel_;
            modelparams.nmodeloutputchannel_ = NULL;
        }
        // 释放输出宽度
        if (NULL != modelparams.nmodeloutputweith_)
        {
            delete[] modelparams.nmodeloutputweith_;
            modelparams.nmodeloutputweith_ = NULL;
        }

        // 释放输出高度
        if (NULL != modelparams.nmodeloutputheight_)
        {
            delete[] modelparams.nmodeloutputheight_;
            modelparams.nmodeloutputheight_ = NULL;
        }


        if (NULL != resize_image_ || NULL != input_data_)
        {   
            for (size_t imgunm = 0; imgunm < modelparams.io_num.n_input; imgunm++)
            {
                if (!resize_image_[imgunm].empty())
                {
                    resize_image_[imgunm].release();
                }
            }
            if (NULL != resize_image_)
            {
                delete[] resize_image_;
                resize_image_ = NULL;
            }       
            if (NULL != input_data_)
            {
                delete[] input_data_;
                input_data_ = NULL;
            }
        }

        // 释放申请的内存　
        if (NULL != ecogroundobjectdects_.ecogroundobject)
        {
            for (size_t nobj = 0; nobj < nmaxdetectnum_; nobj++)
            {
                if (NULL != ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.ptrecogroundobjectscls)
                {
                    delete[] ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.ptrecogroundobjectscls;
                    ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.ptrecogroundobjectscls = NULL;
                    ecogroundobjectdects_.ecogroundobject[nobj].groundobjectsCls.nobjectsclsnum = -1;
                }
            }
            
            delete[] ecogroundobjectdects_.ecogroundobject;
            ecogroundobjectdects_.ecogroundobject = NULL;
            ecogroundobjectdects_.ngroundobjectnum = -1;
        }

        // 零拷贝申请的输入、输出内存释放
        if(modelparams.ctx > 0) 
        {
            for (int i = 0; i < modelparams.io_num.n_input; i++) {
                if (modelparams.input_mems[i] != NULL) {
                    int ret = rknn_destroy_mem(modelparams.ctx, modelparams.input_mems[i]);
                    if (ret != RKNN_SUCC) {
                        printf("rknn_destroy_mem fail! ret=%d\n", ret);
                    }
                }
            }
            for (int i = 0; i < modelparams.io_num.n_output; i++) {
                if (modelparams.output_mems[i] != NULL) {
                    int ret = rknn_destroy_mem(modelparams.ctx, modelparams.output_mems[i]);
                    if (ret != RKNN_SUCC) {
                        printf("rknn_destroy_mem fail! ret=%d\n", ret);
                    }
                }
            }
        }

        // Release
        if(modelparams.ctx > 0) 
        {
            rknn_destroy(modelparams.ctx);
            modelparams.ctx = 0;
        }

        if (NULL != modelparams.model_data)
        {
            free(modelparams.model_data);
            modelparams.model_data = NULL;
        }
        
        return EStatus_Success;
    }


    /// 在detect_self.cc中用的
    // EcoEStatus EcoDetectInference::draw_objects(const cv::Mat& bgr)
    EcoEStatus EcoDetectInference::draw_objects(const cv::Mat& bgr, const EcoAInterfaceDeebotStatus_t& st, std::string image_path, cv::Mat& image_before)
    {
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型

        if (save_img_result_ == 0)
        {
            std::cout << "save_img_result_ == 0 can‘t save image result" << std::endl;
            return ecoestatus_;
        }
        
        if (bgr.empty())
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout  <<  "input img is empty:" <<  ecoestatus_ <<std::endl;
            return ecoestatus_;
        }

        cv::Mat image = bgr.clone();
        
        if(image.channels() > 1)
        {
            if (detecttype_ == EM_YOLOV8_IR_STAIN)
            {
                cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_LINEAR);
            }
            else
            {
                cv::resize(image, image, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
            }
        }
        
        char text[256];

        for (size_t i = 0; i < ecogroundobjectdects_.ngroundobjectnum; i++)
        {
            EcoGroundObjectCls*  objlabel = &ecogroundobjectdects_.ecogroundobject[i].groundobjectsCls.ptrecogroundobjectscls[0];
            EcoGroundObjectDect* obj      = &ecogroundobjectdects_.ecogroundobject[i];

            if(!obj->bisobjects)
            {
                continue;
            }

            ////这里的x1,y1是左上点，x2和y2是右下点
            int x1 = obj->rect.x;
            int y1 = obj->rect.y;
            int x2 = obj->rect.x + obj->rect.width;
            int y2 = obj->rect.y + obj->rect.height;
            std::cout << x1  << "  " << y1  << "  "  << x2   << "  "   << y2 << std::endl;

            rectangle(image, obj->rect, cv::Scalar(255, 0, 0), 4);
            sprintf(text, "%d = %.5f", (int)objlabel->label, objlabel->fconfidence);
            putText(image, text, cv::Point(x1, y1 + 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, vcolor_inner[objlabel->inlabel % 20]);

            std::cout <<"label = " << (int)objlabel->label <<"  position[0].x = " << (int)obj->position[0].x << "  position[0].y = " << (int)obj->position[0].y << std::endl;
            sprintf(text, "%d %d", (int)obj->position[0].x, (int)obj->position[0].y);
            putText(image, text, cv::Point(x1, y2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, vcolor_inner[objlabel->inlabel % 20]);

            std::cout <<"label = " << (int)objlabel->label << "position[1].x = " << (int)obj->position[1].x << "  position[1].y = " << (int)obj->position[1].y << std::endl;
            sprintf(text, "%d %d", (int)obj->position[1].x, (int)obj->position[1].y);
            putText(image, text, cv::Point(x2, y2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, vcolor_inner[objlabel->inlabel % 20]);

        }

        if(-1 != access(image_path.c_str(), 0))
        {
            if(image.channels() > 1)
            {
                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
            }
            
            std::string str_name = image_path + std::to_string(st.timestamp) + "_" + std::to_string(st.x) +
            "_" + std::to_string(st.y) + "_" + std::to_string(st.Qz) + "_" + std::to_string(ecogroundobjectdects_.ngroundobjectnum) + ".jpg";
            std::cout << "str_name = " << str_name << std::endl;
            imwrite(str_name, image);
        }
       
        return EStatus_Success;
    }

}




