
/******************************************************************************
模块名　　　　：　eco_ai_cls
文件名　　　　：　eco_ai_cls.cpp
相关文件　　　：　eco_ai_cls.cpp
文件实现功能　：　分类识别(特征提取)函数定义
作者　　　　　：　周峰
版本　　　　　：　1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        创建人    修改内容
2022/01/08    1.0                     周峰         创建
******************************************************************************/

#include"utils.h"
#include"postprocess.h"
#include"eco_ai_cls.h"


namespace sweeper_ai
{

    EcoObjectClsInference::EcoObjectClsInference():
    bOpenFlag(false), resize_image_(NULL), input_data_(NULL), groundobjectscls_(EcoGroundObjectsCls()), ecoextractblobs_(EcoExtractBlobs()),
    ecoaiclstypes_(EM_CLS_NONE), ntopkcls_(0), nextranum_(0), fclsThreshold_(0.0),
    background_(-1), modelparams(EcoRknnModelParams()), resize_type(NONE_RESIZE)
    {

    }


    EcoObjectClsInference::~EcoObjectClsInference()
    {

    }


    EcoEStatus EcoObjectClsInference::ecoObjectClsOpen(rapidjson::Value &param)
    {
        int ret(0);
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型
        //判断配置文件是否存在
        if (param.IsNull())
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "objclsparam.IsNull() in EcoObjectClsInference::ecoObjectClsOpen"  << std::endl;
            return ecoestatus_;
        }

        if (param.HasMember("model_type"))
        {
            ecoaiclstypes_ = EcoAIClsTypes(param["model_type"].GetInt());                //　分类以及特征提取定义
        }

        if (param.HasMember("background"))
        {
            background_ = param["background"].GetInt();                                 //　背景类标签
        }

        if (param.HasMember("resize_type"))
        {
            resize_type = (EcoResizeTypeS)(param["resize_type"].GetInt());         //　图像预处理方式，resize（0） 还是 pad + resize（1）
            std::cout << "resize_type = " << resize_type << std::endl;
        }


        if (param.HasMember("cls_topk"))
        {
            ntopkcls_  = param["cls_topk"].GetInt();                                     //　输出分类前　ｔｏｐｋ　类
        }

        if (param.HasMember("extra_num"))
        {
            nextranum_  = param["extra_num"].GetInt();                                     //　输出提取的特征ｂｌｏｂ个数
        }

        int model_data_size = 0;
        if (param.HasMember("name"))
        {
            std::string model_name_ = param["name"].GetString();    //　目标检测模型结构文件

            //判断配置文件是否存在
            if(-1 == access(model_name_.c_str(), 0))
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "have no model_file: " << model_name_ << std::endl;
                ecoObjectClsClose();
                return ecoestatus_;
            }

            //　读取算法模型　
            modelparams.model_data = load_model(model_name_.c_str(), &model_data_size);
            if (NULL == modelparams.model_data)
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "load_model error " << std::endl;
                ecoObjectClsClose();
                return ecoestatus_;
            }
        }        
        else
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "have no model file"<< std::endl;
            ecoObjectClsClose();
            return ecoestatus_;
        }

        //　ctx 环境句柄初始化
        ret = rknn_init(&modelparams.ctx, modelparams.model_data, model_data_size, 0, NULL);    
        if (ret < 0)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_init error ret = " << ret << std::endl;
            ecoObjectClsClose();
            return ecoestatus_;
        }

        //　获取设定的npu_id并设置模型所在npu
        rknn_core_mask core_mask = rknn_core_mask(param["core"].GetInt());
        if (core_mask < RKNN_NPU_CORE_AUTO || core_mask > RKNN_NPU_CORE_0_1_2)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "NPU_ID is ERROR npu_id　=　" << param["core"].GetInt() << std::endl;
            ecoObjectClsClose();
            return ecoestatus_;
        }

        //　设定模型运行　ｎｐｕ       
        ret = rknn_set_core_mask(modelparams.ctx, core_mask);
        if (ret < 0)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_set_core_mask error ret =　" << ret << std::endl;
            ecoObjectClsClose();
            return ecoestatus_;
        }

        //　获取模型输入和输出节点个数                    
        ret = rknn_query(modelparams.ctx, RKNN_QUERY_IN_OUT_NUM, &modelparams.io_num, sizeof(modelparams.io_num));
        if (ret < 0)
        {
            printf("rknn_query rknn_input_output_num error ret = %d\n", ret);
            ecoObjectClsClose();
            return ecoestatus_;
        }

        modelparams.nmodelinputchannel_ = new int[modelparams.io_num.n_input];
        modelparams.nmodelinputweith_   = new int[modelparams.io_num.n_input];
        modelparams.nmodelinputheight_  = new int[modelparams.io_num.n_input];
        resize_image_      = new cv::Mat[modelparams.io_num.n_input];

        modelparams.nmodeloutputchannel_ = new int[modelparams.io_num.n_output];                           // 网络输入　宽度
        modelparams.nmodeloutputweith_   = new int[modelparams.io_num.n_output];                           // 网络输入　宽度
        modelparams.nmodeloutputheight_  = new int[modelparams.io_num.n_output];                           // 网络输入　高度

        input_data_        = new int8_t[MAX_MALLOC_NUM];

        if (NULL == modelparams.nmodelinputchannel_   || NULL == modelparams.nmodelinputweith_   || NULL == modelparams.nmodelinputheight_ || NULL == resize_image_ 
             || NULL == input_data_ || NULL == modelparams.nmodeloutputchannel_ || NULL == modelparams.nmodeloutputweith_ || NULL == modelparams.nmodeloutputheight_)
        {
            ecoestatus_ = EStatus_OutOfMemory;
            std::cout  <<  "NULL == nmodelinputchannel_ || NULL == nmodelinputweith_ || NULL == nmodelinputheight_ || NULL == resize_image_ || NULL == input_data_ can't memory, ecoestatus_ = " <<  ecoestatus_ <<std::endl;
            ecoObjectClsClose();
            return ecoestatus_;
        }

        //　从句柄中获取　输入节点参数
        ret = rknn_get_ctx_attr(modelparams);
        if (ret < 0)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_get_ctx_attr error ret =　" << ret << std::endl;
            ecoObjectClsClose();
            return ecoestatus_;
        }

        for (size_t imginput = 0; imginput < modelparams.io_num.n_input; imginput++)
        {
            resize_image_[imginput] = cv::Mat(modelparams.nmodelinputheight_[imginput], modelparams.nmodelinputweith_[imginput],  CV_8UC3, cv::Scalar(114, 114, 114));
        }

        // 申请分类输出所需内存大小    
        if ((ecoaiclstypes_ == EM_CLS || ecoaiclstypes_ == EM_BOTH_CLS) && ntopkcls_ > 0)
        {
            //模型 ID 编号
            if (param.HasMember("id"))
            {
                groundobjectscls_.model_id = param["id"].GetInt();                //　分类以及特征提取定义
            }
            groundobjectscls_.nobjectsclsnum = ntopkcls_;
            groundobjectscls_.ptrecogroundobjectscls = new EcoGroundObjectCls[ntopkcls_];
            if (NULL == groundobjectscls_.ptrecogroundobjectscls)
            {
                ecoestatus_ = EStatus_OutOfMemory;
                std::cout  <<  "groundobjectscls_.ptrecogroundobjectscls can't memory" <<  ecoestatus_ <<std::endl;
                ecoObjectClsClose();
                return ecoestatus_;
            }
            memset(groundobjectscls_.ptrecogroundobjectscls, 0, ntopkcls_ * sizeof(EcoGroundObjectCls));
        }
        // 申请特征提取输出所需内存大小   
        if ((ecoaiclstypes_ == EM_EXTRACTBLOBS || ecoaiclstypes_== EM_BOTH_CLS) && nextranum_ > 0)
        {
            //模型 ID 编号
            if (param.HasMember("id"))
            {
                ecoextractblobs_.model_id = param["id"].GetInt();                //　分类以及特征提取定义
            }
            ecoextractblobs_.nblobsnum = nextranum_;
            ecoextractblobs_.ptrecoextractblob = new EcoExtractBlob[nextranum_];
            if (NULL == ecoextractblobs_.ptrecoextractblob)
            {
                ecoestatus_ = EStatus_OutOfMemory;
                std::cout  <<  "ecoextractblobs_.ptrecoextractblob can't memory, ecoestatus_ = " <<  ecoestatus_ <<std::endl;
                ecoObjectClsClose();
                return ecoestatus_;
            }
            memset(ecoextractblobs_.ptrecoextractblob, 0, nextranum_ * sizeof(EcoExtractBlob));

            for (int ib = 0; ib < nextranum_; ib++)
            {
                ecoextractblobs_.ptrecoextractblob[ib].blob = new float[MAX_EXTRO_BLOB_LEN];
                if (NULL == ecoextractblobs_.ptrecoextractblob[ib].blob)
                {
                    ecoestatus_ = EStatus_OutOfMemory;
                    std::cout  <<  "ecoextractblobs_.ptrecoextractblob[" << ib << "].blob can't memory, ecoestatus_ = " <<  ecoestatus_ <<std::endl;
                    ecoObjectClsClose();
                    return ecoestatus_;
                }
                memset(ecoextractblobs_.ptrecoextractblob[ib].blob, 0, sizeof(float) * MAX_EXTRO_BLOB_LEN);
            }    
        }

        bOpenFlag = true;
        return EStatus_Success;

    }




    EcoEStatus EcoObjectClsInference::ecoObjectClsInfer(const std::vector<cv::Mat> &img,const cv::Rect &ROI_rect)
    {
        int ret(0);
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型
        std::vector<float> cls_scores;

        if (img.empty() || img.size() != modelparams.io_num.n_input)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout  <<  "img.empty() || img.size():" << img.size() << " != io_num.n_input:" << modelparams.io_num.n_input << " in EcoObjectClsInference::ecoObjectClsInfer" <<  ecoestatus_ << std::endl;
            return ecoestatus_;
        }

        //  set inputs 设置模型输入参数
        rknn_input inputs[modelparams.io_num.n_input];
        memset(inputs, 0, sizeof(inputs));

        for (int i = 0; i < modelparams.io_num.n_input; i++)
        {
            if (img[i].empty())
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout  <<  "input img[ " << i << " ] is empty in EcoObjectClsInference::ecoObjectClsInfer" <<  ecoestatus_ << std::endl;
                return ecoestatus_;
            }   
                
            cv::Mat bgr = img[i](ROI_rect);

            inputs[i].index = i;
            inputs[i].type  = RKNN_TENSOR_UINT8;
            inputs[i].size  = modelparams.nmodelinputweith_[i] * modelparams.nmodelinputheight_[i] * modelparams.nmodelinputchannel_[i];
            inputs[i].fmt   = RKNN_TENSOR_NHWC;
            inputs[i].pass_through = 0;

            int * pu8ImgData = NULL;
            //　resize image
            if(bgr.cols != modelparams.nmodelinputweith_[i] || bgr.rows != modelparams.nmodelinputheight_[i]) 
            {
                eco_resize(bgr, resize_image_[i], modelparams.nmodelinputweith_[i], modelparams.nmodelinputheight_[i], resize_type);
                
                pu8ImgData = resize_image_[i].ptr<int>(0);    
            }
            else
            {
                pu8ImgData = bgr.ptr<int>(0); 
            }

            inputs[i].buf = (void *)pu8ImgData;   
        }

        //  Get outputs　获取模型输出
        rknn_output outputs[modelparams.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < modelparams.io_num.n_output; i++)
        {
            outputs[i].index = i;
            outputs[i].is_prealloc = 0;
            outputs[i].want_float = 0;
        }


        //　图像数据传入模型
        ret = rknn_inputs_set(modelparams.ctx, modelparams.io_num.n_input, inputs);
        if(ret < 0) 
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_input_set fail! ret =　" << ret << std::endl;
            return ecoestatus_;
        }

        // Run
        ret = rknn_run(modelparams.ctx, nullptr);
        if(ret < 0) 
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_run fail! ret =　" << ret << std::endl;
            return ecoestatus_;
        }

        //　获取模型处理结果
        ret = rknn_outputs_get(modelparams.ctx, modelparams.io_num.n_output, outputs, NULL);
        if(ret < 0) 
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_outputs_get fail! ret =　" << ret << std::endl;
            return ecoestatus_;
        }

        // Post Process
        if (ecoaiclstypes_ == EM_CLS || ecoaiclstypes_ == EM_BOTH_CLS)
        {
            for (int i = 0; i < modelparams.io_num.n_output; i++)
            {
                int   MaxClass[ntopkcls_];
                int   sz = outputs[i].size / (ntopkcls_ - 1);
                float fMaxProb[ntopkcls_];
                float *buffer = (float *)outputs[i].buf;

                //　获取　ｔｏｐｋ的分类结果    
                rknn_cls_GetTop(background_, buffer, fMaxProb, MaxClass, sz, ntopkcls_);

                printf(" --- Top_ntopkcls_ ---\n");
                for(int i = 0; i < ntopkcls_; i++)
                {
                    groundobjectscls_.ptrecogroundobjectscls[i].fconfidence = fMaxProb[i];

                    groundobjectscls_.ptrecogroundobjectscls[i].inlabel = MaxClass[i];
                    printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
                }
            }
        }
        else if (ecoaiclstypes_ == EM_EXTRACTBLOBS || ecoaiclstypes_ == EM_BOTH_CLS)
        {
            for (int i = 0; i < modelparams.io_num.n_output; i++)
            {
                float *buffer = (float *)outputs[i].buf;

                memcpy(ecoextractblobs_.ptrecoextractblob[i].blob, buffer, sizeof(float) * MAX_EXTRO_BLOB_LEN);

            }
        }
        
        // Release rknn_outputs
        //　释放输出节点内存
        ret = rknn_outputs_release(modelparams.ctx, modelparams.io_num.n_output, outputs);
        if(ret < 0) 
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_outputs_get fail! ret =　" << ret << std::endl;
            return ecoestatus_;
        }
        return ecoestatus_;
    }


    EcoEStatus EcoObjectClsInference::ecoObjectClsClose()
    {
        EcoEStatus ecoestatus_(EStatus_Success);
        // 释放输入通道数
        if (NULL != modelparams.nmodelinputchannel_)
        {
            delete[] modelparams.nmodelinputchannel_;
            modelparams.nmodelinputchannel_ = NULL;
        }
        // 释放输入宽度
        if (NULL != modelparams.nmodelinputweith_)
        {
            delete[] modelparams.nmodelinputweith_;
            modelparams.nmodelinputweith_ = NULL;
        }
        // 释放输入高度
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

        //释放分类的结果内存
        if (NULL != groundobjectscls_.ptrecogroundobjectscls) 
        {
            delete[] groundobjectscls_.ptrecogroundobjectscls;
            groundobjectscls_.ptrecogroundobjectscls = NULL;
        }

        //　释放特征提取网络
        if(NULL != ecoextractblobs_.ptrecoextractblob) 
        {
            for (int ib = 0; ib < ecoextractblobs_.nblobsnum; ib++)
            {
                if (NULL != ecoextractblobs_.ptrecoextractblob[ib].blob)
                {
                    delete[] ecoextractblobs_.ptrecoextractblob[ib].blob;
                    ecoextractblobs_.ptrecoextractblob[ib].blob = NULL;
                }
            }

            delete[] ecoextractblobs_.ptrecoextractblob;
            ecoextractblobs_.ptrecoextractblob = NULL;
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
        return ecoestatus_;
    }


}