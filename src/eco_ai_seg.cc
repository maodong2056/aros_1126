/******************************************************************************
模块名　　　　： eco_ai_seg.h
文件名　　　　： eco_ai_seg.cc
相关文件　　　： eco_ai_seg.cc
文件实现功能　： 语义分割类源码
作者　　　　　： 周峰
版本　　　　　： 1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        创建人    修改内容
2022/01/08    1.0                     周峰      创建
2022/10/08    1.1        周峰                   添加分割后处理
******************************************************************************/

#include"postprocess.h"
#include"eco_ai_seg.h"
#include"utils.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <set>
#include <string>
#include <fstream>

static std::vector<cv::Scalar> vcolor_inner={
      cv::Scalar(0, 0, 125)  , cv::Scalar(0, 125, 0)   , cv::Scalar(125, 0, 0)
    , cv::Scalar(0, 0, 255)  , cv::Scalar(0, 225, 0)   , cv::Scalar(255, 0, 0)
    , cv::Scalar(125, 125, 0), cv::Scalar(125, 0, 125) , cv::Scalar(0, 125, 125)
    , cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255) , cv::Scalar(0, 255, 255)};

namespace sweeper_ai
{

    EcoSegInference::EcoSegInference():
    bOpenFlag(false), poutputbuf(NULL), resize_image_(NULL),input_data_(NULL),
    clsthreshold_(0.5), iouThreshold_(0.45), linelaser_ground_points_colmean_(3,0.0),ecoinstanceobjectseg(EcoInstanceObjectSeg()), 
    ecoaisegtypes_(EM_SEG_NONE),background_(-1), modelparams(EcoRknnModelParams()), 
    resize_type(NONE_RESIZE)
    {
        out_threshold_of_all.clear();
        after_model_ids.clear();
    }

    EcoSegInference::~EcoSegInference()
    {

    }

    EcoEStatus EcoSegInference::ecoSegOpen(rapidjson::Value &param)
    {
        int ret(0);
        EcoEStatus ecoestatus_(EStatus_Success);

        out_threshold_of_all.clear();  //每个类的阈值
        poutputbuf = NULL;

        if (param.IsNull())
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "segparam.IsNull() in EcoSegInference::ecoSegOpen"  << std::endl;
            return ecoestatus_;
        }

        if (param.HasMember("cls_threshold"))
        {
           clsthreshold_ = param["cls_threshold"].GetFloat();                    //　输出提取的特征ｂｌｏｂ个数
        }

        if (param.HasMember("iou_threshold"))
        {
            iouThreshold_ = param["iou_threshold"].GetFloat();               //　针对地毯模型既有检测又有分割模型设置的IOU阈值
        }

        if (param.HasMember("model_type"))
        {
            ecoaisegtypes_  = (EM_EcoAISegTypes)(param["model_type"].GetInt());
            std::cout << "ecoaisegtypes_ = " << (int)ecoaisegtypes_ << std::endl;
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

        if (param.HasMember("threshold"))
        {
            //　定制化每类阈值
            rapidjson::Value threshold_of_each = param["threshold"].GetArray();   
            for (rapidjson::SizeType j = 0; j < threshold_of_each.Size(); j++)
            {
                std::vector<float> out_threshold_of_each_;
                for(rapidjson::SizeType i = 0; i < threshold_of_each[j].Size(); i++){
                    out_threshold_of_each_.push_back(threshold_of_each[j][i].GetFloat());
                }
                out_threshold_of_all.push_back(out_threshold_of_each_);
            }  
        }

        if (param.HasMember("after_model"))
        {
            //　定制化每类阈值
            rapidjson::Value after_models = param["after_model"].GetArray();   
            std::cout << "the number of after_model is " << after_models.Size() << std::endl << "the threshold of each:";
            
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
                    // 创建文件夹
                    int result = mkdir(image_save_path.c_str(), 0755);
                
                    if (result == 0) 
                    {
                        std::cout << "Folder created successfully." << std::endl;
                    } 
                    else 
                    {
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
            
            int model_data_size = 0;  // 模型内存大小

            std::string model_name_ = param["name"].GetString();    //　语义分割模型结构文件

            int model_id = param["id"].GetInt();
            if (model_id == 3)
            {
                int country_id = 0;
                std::ifstream file("/tmp/country.bin");
                if (!file.is_open()) {
                    std::cout << "open country.bin error" << std::endl;
                }
                else
                {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string jsonStr = buffer.str();
                    file.close();

                    rapidjson::Document country_doc;
                    country_doc.Parse(jsonStr.c_str());

                    if (country_doc.HasParseError()) {
                        std::cout << "country json error " << country_doc.GetParseError() << std::endl;
                    }
                    else
                    {
                        if (country_doc.IsObject() && country_doc.HasMember("country") && 
                            country_doc["country"].IsString()) 
                        {
                            const rapidjson::Value& country = country_doc["country"];
                            std::string country_str = country.GetString();
                            
                            if (country_str == "ZH") {
                                country_id = 0;
                            } else {
                                country_id = 1;
                            }
                            std::cout << "country: " << country_str << "; country_id: " << country_id << std::endl;
                        }
                    }
                    if (country_id == 1)
                    {
                        std::string file_name = model_name_;
                        size_t last_dot_pos = file_name.find_last_of(".");
                        if (last_dot_pos != std::string::npos) {
                            file_name.insert(last_dot_pos, "-oversea");
                            if(-1 == access(file_name.c_str(), 0))
                            {
                                std::cout << "have no oversea model_file " << file_name << std::endl;
                            }
                            else
                            {
                                model_name_ = file_name;
                            }
                        }
                    }
                }

            }

            std::cout << "model_name_ = " << model_name_ << std::endl;
            //判断配置文件是否存在
            if(-1 == access(model_name_.c_str(), 0))
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "have no model_file " << model_name_ << std::endl;
                ecoSegClose();
                return ecoestatus_;
            }

            modelparams.model_data = load_model(model_name_.c_str(), &model_data_size);
            if (NULL == modelparams.model_data)
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout << "load_model error " << std::endl;
                ecoSegClose();
                return ecoestatus_;
            }

            ret = rknn_init(&modelparams.ctx, modelparams.model_data, model_data_size, 0, NULL);    //　ctx　环境句柄初始化
            if (ret < 0)
            {
                ecoestatus_ = EStatus_InvalidParameter;
                printf("rknn_init error ret = %d\n", ret);
                ecoSegClose();
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
            ecoSegClose();
            return ecoestatus_;
        }

        //获取设定的npu_id并设置模型所在npu
        rknn_core_mask core_mask = rknn_core_mask(param["core"].GetInt());
        if (core_mask< RKNN_NPU_CORE_AUTO || core_mask > RKNN_NPU_CORE_0_1_2)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            printf("NPU_ID is ERROR npu_id = %d\n", param["core"].GetInt());
            ecoSegClose();
            return ecoestatus_;
        }

        // //设定模型运行　ｎｐｕ       
        // ret = rknn_set_core_mask(modelparams.ctx, core_mask);
        // if (ret < 0)
        // {
        //     ecoestatus_ = EStatus_InvalidParameter;
        //     printf("rknn_set_core_mask error ret=%d\n", ret);
        //     ecoSegClose();
        //     return ecoestatus_;
        // }

        //　获取模型输入和输出节点个数                    
        ret = rknn_query(modelparams.ctx, RKNN_QUERY_IN_OUT_NUM, &modelparams.io_num, sizeof(modelparams.io_num));
        if (ret < 0)
        {
            printf("rknn_query rknn_input_output_num error ret = %d\n", ret);
            ecoSegClose();
            return ecoestatus_;
        }

        // 网络输入节点　通道数，宽度，高度
        poutputbuf         = new int8_t* [3];  
        modelparams.nmodelinputchannel_ = new int[modelparams.io_num.n_input];
        modelparams.nmodelinputweith_   = new int[modelparams.io_num.n_input];
        modelparams.nmodelinputheight_  = new int[modelparams.io_num.n_input];

        // 网络输入图片内存
        resize_image_      = new cv::Mat[modelparams.io_num.n_input];

        // 网络输出节点　通道数，宽度，高度
        modelparams.nmodeloutputchannel_ = new int[modelparams.io_num.n_output];
        modelparams.nmodeloutputweith_   = new int[modelparams.io_num.n_output];
        modelparams.nmodeloutputheight_  = new int[modelparams.io_num.n_output];

        input_data_        = new int8_t[MAX_MALLOC_NUM];


        if (NULL == poutputbuf  || NULL == modelparams.nmodelinputchannel_ || NULL == modelparams.nmodelinputweith_ 
        || NULL == modelparams.nmodelinputheight_ || NULL == resize_image_ 
             || NULL == input_data_ || NULL == modelparams.nmodeloutputchannel_ 
             || NULL == modelparams.nmodeloutputweith_ || NULL == modelparams.nmodeloutputheight_)
        {
            ecoestatus_ = EStatus_OutOfMemory;
            std::cout  <<  "NULL == nmodelinputchannel_ || NULL == nmodelinputweith_ || NULL == nmodelinputheight_ || NULL == resize_image_ || NULL == input_data_ can't memory, ecoestatus_ = " <<  ecoestatus_ <<std::endl;
            ecoSegClose();
            return ecoestatus_;
        }

        for (size_t outnum = 0; outnum < 3; outnum++)
        {
            poutputbuf[outnum] = NULL;
        }

        //　从句柄中获取　输入节点参数
        // 零拷贝修改rknn_get_ctx_attr，添加输入输出的内存申请
        // ret = rknn_get_ctx_attr(modelparams);
        std::cout << "----------use zero copy---------" << std::endl;
        ret = rknn_get_ctx_attr_zero_copy(modelparams); 
        if (ret < 0)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout << "rknn_get_ctx_attr error ret = " << ret << std::endl;
            ecoSegClose();
            return ecoestatus_;
        }

        for (size_t imginput = 0; imginput < modelparams.io_num.n_input; imginput++)
        {
            resize_image_[imginput] = cv::Mat(modelparams.nmodelinputheight_[imginput], modelparams.nmodelinputweith_[imginput],  CV_8UC3, cv::Scalar(114, 114, 114));
        }

        //模型 ID 编号
        if (param.HasMember("id"))
        {
            ecoinstanceobjectseg.model_id = param["id"].GetInt();                //　分类以及特征提取定义
        }

        if(ecoinstanceobjectseg.model_id == 3)
        {
            ecoinstanceobjectseg.mask = cv::Mat(modelparams.nmodelinputheight_[0], modelparams.nmodelinputweith_[0], CV_8UC1, cv::Scalar(255));
        }
        
        ecoinstanceobjectseg.maskdata.clear();

        bOpenFlag =true;

        return ecoestatus_;
    }

    EcoEStatus EcoSegInference::ecoSegInfer(const std::vector<cv::Mat> &img, const cv::Rect &ROI_rect, int& cm_distance)
    {

        int ret(0);
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型

        if (img.empty() || img.size() < modelparams.io_num.n_input)
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout  <<  "img.empty() || img.size():" << img.size() << " < io_num.n_input:" << modelparams.io_num.n_input << " in EcoSegInference::ecoSegInfer" <<  ecoestatus_ << std::endl;
            return ecoestatus_;
        }

        // 零拷贝省略rknn_input
        for (size_t i = 0; i < modelparams.io_num.n_input; i++)
        {
            if (img[i].empty())
            {
                ecoestatus_ = EStatus_InvalidParameter;
                std::cout  <<  "input img[ " << i << " ] is empty in EcoSegInference::ecoSegInfer" <<  ecoestatus_ << std::endl;
                return ecoestatus_;
            }
            
            cv::Mat bgr = img[i](ROI_rect);

            //　resize image   输入图像大小变化
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
        if(EM_freespace_seg == ecoaisegtypes_)
        {
            want_float = 1;
        }
        else if(EM_POINTS_DETECT == ecoaisegtypes_)
        {
            want_float = 0;
        }

        // Run
        ret = rknn_run(modelparams.ctx, nullptr);
        if(ret < 0) 
        {
            ecoestatus_ = EStatus_InvalidParameter;
            printf("rknn_run fail! ret=%d\n", ret);
            return ecoestatus_;
        }
        // if(EM_freespace_seg == ecoaisegtypes_)
        // {
        //     //　获取当前时间并计算运行时间
        //     rknn_perf_run perf_run;
        //     ret = rknn_query(modelparams.ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
        //     std::cout << "EM_freespace_seg  rknn_run time =" << perf_run.run_duration/1000.0 << "ms" << std::endl; 
        // }

        // if(EM_POINTS_DETECT == ecoaisegtypes_)
        // {
        //     //　获取当前时间并计算运行时间
        //     rknn_perf_run perf_run;
        //     ret = rknn_query(modelparams.ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
        //     std::cout << "EM_POINTS_DETECT  rknn_run time =" << perf_run.run_duration/1000.0 << "ms" << std::endl; 
        // }

#ifdef D_DEBUG
        //　获取当前时间并计算运行时间
        rknn_perf_run perf_run;
        ret = rknn_query(modelparams.ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
        std::cout << "rknn_run time =" << perf_run.run_duration/1000.0 << "ms" << std::endl; 
#endif

#ifdef D_DEBUG
    auto end2 = std::chrono::system_clock::now();
#endif

/******** EM_freespace_seg ***********************************************************************************************/ 
        if(EM_freespace_seg == ecoaisegtypes_)
        {
            std::cout<<"start freespace_postprocess"<<std::endl;
            // auto end2 = std::chrono::system_clock::now();
            // int image_in_w = img[0].cols;           ////这里得到的image_in_w是1280
            // int image_in_h = img[0].rows;           ////这里得到的image_in_h是960

            int image_in_w = 1280;           ////把输入进来的512*384大小图片强制改成1280*960
            int image_in_h = 960;           

            ecoinstanceobjectseg.maskdata.clear();


            // ecoinstanceobjectseg.mask.setTo(cv::Scalar(0));

            if (!ecoinstanceobjectseg.mask.empty())
            {
                ecoinstanceobjectseg.mask.release();
                ecoinstanceobjectseg.mask = cv::Mat(192, 256, CV_8UC1, cv::Scalar(0));
            }

            // freespace_postprocess(modelparams, outputs, ecoinstanceobjectseg.mask, ecoinstanceobjectseg.maskdata, image_in_w, image_in_h, clsthreshold_, 
            //          iouThreshold_, out_threshold_of_all, cm_distance);
            // 零拷贝
            freespace_postprocess_zero_copy(modelparams, modelparams.output_mems, ecoinstanceobjectseg.mask, ecoinstanceobjectseg.maskdata, image_in_w, image_in_h, clsthreshold_, 
                        iouThreshold_, out_threshold_of_all, cm_distance);
            std::cout << "stop freespace_postprocess, maskdata.size() = " << ecoinstanceobjectseg.maskdata.size()<< std::endl;
            // auto end4 = std::chrono::system_clock::now();
            // auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end2);
            // std::cout << "freespace_postprocess post  rknn_run time =" << duration4.count() << "ms" << std::endl;

        }
        else if(EM_POINTS_DETECT == ecoaisegtypes_)
        {
            std::cout << "start groundpoint_postprocess" << std::endl;
            // auto end2 = std::chrono::system_clock::now();
            int image_in_w = 1280;           ////把输入进来的512*384大小图片强制改成1280*960
            int image_in_h = 960;           
            ecoinstanceobjectseg.maskdata.clear();

            // groundpoint_postprocess(modelparams, outputs, ecoinstanceobjectseg.maskdata, image_in_w, image_in_h,  0.5, cm_distance);
            // 零拷贝
            groundpoint_postprocess_zero_copy(modelparams, modelparams.output_mems, ecoinstanceobjectseg.maskdata, image_in_w, image_in_h,  0.5, cm_distance);
            std::cout << "stop groundpoint_postprocess: " << ecoinstanceobjectseg.maskdata.size() << std::endl;
            // auto end4 = std::chrono::system_clock::now();
            // auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end2);
            // std::cout << "EM_POINTS_DETECT post  rknn_run time =" << duration4.count() << "ms" << std::endl;

        }

      return ecoestatus_;

    }


    EcoEStatus EcoSegInference::ecoSegClose()
    {

        EcoEStatus ecoestatus_(EStatus_Success);   

        if (NULL != poutputbuf)
        {
            delete[] poutputbuf;
            poutputbuf = NULL;
        }
        
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

        if (!ecoinstanceobjectseg.mask.empty())
        {
            ecoinstanceobjectseg.mask.release();
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

        return ecoestatus_;
    }



    // EcoEStatus EcoSegInference::draw_objects(const std::vector<EcoKeyPoint>& segmaskdata, const cv::Mat& bgr, const EcoAInterfaceDeebotStatus_t& st, std::string image_path)
    // {
    //     EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型

    //     if (bgr.empty())
    //     {
    //         ecoestatus_ = EStatus_InvalidParameter;
    //         std::cout  <<  "input img is empty:" <<  ecoestatus_ <<std::endl;
    //         return ecoestatus_;
    //     }

    //     cv::Mat image = bgr.clone();
    //     cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    //     cv::resize(image, image, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
    //     char text[256];

    //     if (segmaskdata.size() > 0)
    //     {
    //         for (size_t kk = 0; kk < segmaskdata.size(); kk++)
    //         {
    //             cv::Point2f points = segmaskdata[kk].mappos;
    //             if(int(segmaskdata[kk].label) != 999 && segmaskdata[kk].bistrue
    //             && points.x > 0 && points.x < image.cols-1 && points.y > 0 && points.y < image.rows -1) 
    //             {
    //                 cv::circle(image, cv::Point2f(points.x, points.y), 3, vcolor_inner[segmaskdata[kk].inlabel % 11], -1);
    //             }
    //         }
    //     }


    //     std::string binary_path_SEG = image_path + std::to_string(st.timestamp) + "_" + std::to_string(st.x) +
    //     "_" + std::to_string(st.y) + "_" + std::to_string(st.Qz) + "_mask_seg.jpg";
    //     cv::imwrite(binary_path_SEG, image);

    //     return EStatus_Success;
    // }

    EcoEStatus EcoSegInference::draw_objects(const std::vector<EcoKeyPoint>& segmaskdata, const cv::Mat& bgr, const EcoAInterfaceDeebotStatus_t& st, std::string image_path)
    {
        EcoEStatus ecoestatus_(EStatus_Success);                             // error　报错类型

        if (bgr.empty())
        {
            ecoestatus_ = EStatus_InvalidParameter;
            std::cout  <<  "input img is empty:" <<  ecoestatus_ <<std::endl;
            return ecoestatus_;
        }

        cv::Mat image = bgr.clone();
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        // cv::resize(image, image, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
        char text[256];

        std::string binary_path_SEG0 = image_path + std::to_string(st.timestamp) + "_" + std::to_string(st.x) +
        "_" + std::to_string(st.y) + "_" + std::to_string(st.Qz) + "_ori_seg.jpg";
        cv::imwrite(binary_path_SEG0, image);

        if (segmaskdata.size() > 0)
        {
            std::set<int> is_colored;
            for (size_t kk = 0; kk < segmaskdata.size(); kk++)
            {
                cv::Point2f points = segmaskdata[kk].mappos;
                points.x = points.x / 2.5;
                points.y = points.y / 2.5;
                if(int(segmaskdata[kk].label) != 999 && segmaskdata[kk].bistrue
                && points.x > 0 && points.x < image.cols-1 && points.y > 0 && points.y < image.rows -1) 
                {
                    // 
                    if (is_colored.find(segmaskdata[kk].inlabel) == is_colored.end()){

                        std::string semantic_text = "";
                        switch(segmaskdata[kk].inlabel){
                            case 2: 
                                semantic_text = "freespace ditan";
                                break;
                            case 4:
                                semantic_text = "freespace uchair";
                                break;
                            case 6:
                                semantic_text = "freespace liusu";
                                break;
                            case 7:
                                semantic_text = "ground point";
                                break;
                            case 8:
                                semantic_text = "ground Line";
                                break;
                            case 9:
                                semantic_text = "ground ditan";
                                break;
                            case 11:
                                semantic_text = "ground menkan";
                                break;
                            case 60:
                                semantic_text = "ground -> liusu";
                                break;
                            case 90:
                                semantic_text = "ground -> ditan";
                                break;
                            case -2:
                                semantic_text = "freespace ditan X";
                                break;
                            case 91:
                                semantic_text = "ground X";
                        }

                        cv::Scalar color;
                        if (segmaskdata[kk].inlabel == 60){
                            color = cv::Scalar(158, 148, 239);  // 障碍物转的流苏点，粉红色
                        }
                        else if (segmaskdata[kk].inlabel == -2){
                            color = cv::Scalar(0, 0, 255);
                        }
                        else if(segmaskdata[kk].inlabel == 90){
                            color = cv::Scalar(18, 246, 238);  // 赋予地毯语义的障碍物点，黄色
                        }
                        else if(segmaskdata[kk].inlabel == 91){
                            color = cv::Scalar(0, 0, 124);  // 无语义且被消掉的障碍物点，深红色
                        }
                        else{
                            color = vcolor_inner[segmaskdata[kk].inlabel];
                        }
                        // cv::Scalar color = segmaskdata[kk].inlabel == 60? cv::Scalar(158, 148, 239): vcolor_inner[segmaskdata[kk].inlabel];
                        cv::putText(image, semantic_text, cv::Point(10, 20 + is_colored.size() * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                        // if (segmaskdata[kk].inlabel == 60){
                        //     cv::putText(image, semantic_text, cv::Point(10, 20 + is_colored.size() * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(158, 148, 239), 2);
                        // }
                        // else{
                            // cv::putText(image, semantic_text, cv::Point(10, 20 + is_colored.size() * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, vcolor_inner[segmaskdata[kk].inlabel], 2);
                        // }
                    }

                    is_colored.insert(segmaskdata[kk].inlabel);

                    cv::Scalar color;
                    if (segmaskdata[kk].inlabel == 60){
                        color = cv::Scalar(158, 148, 239);  // 障碍物转的流苏点，粉红色
                    }
                    else if (segmaskdata[kk].inlabel == -2){
                        color = cv::Scalar(0, 0, 255);
                    }
                    else if(segmaskdata[kk].inlabel == 90){
                        color = cv::Scalar(18, 246, 238);  // 赋予地毯语义的障碍物点，黄色
                    }
                    else if(segmaskdata[kk].inlabel == 91){
                        color = cv::Scalar(0, 0, 124);  // 无语义且被消掉的障碍物点，深红色
                    }
                    else{
                        color = vcolor_inner[segmaskdata[kk].inlabel];
                    }
                    cv::circle(image, cv::Point2f(points.x, points.y), 2, color, -1);

                    // if (segmaskdata[kk].inlabel == 60){
                    //     cv::circle(image, cv::Point2f(points.x, points.y), 3, cv::Scalar(158, 148, 239), -1);
                    // }
                    // else{
                    //     cv::circle(image, cv::Point2f(points.x, points.y), 3, vcolor_inner[segmaskdata[kk].inlabel % 11], -1);
                    // }
                    
                }
            }
        }

        std::string binary_path_SEG = image_path + std::to_string(st.timestamp) + "_" + std::to_string(st.x) +
        "_" + std::to_string(st.y) + "_" + std::to_string(st.Qz) + "_mask_seg.jpg";
        cv::imwrite(binary_path_SEG, image);

        return EStatus_Success;
    }

}
