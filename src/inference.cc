
#include<unistd.h>
#include"inference.h"
#include"eco_task_infer.h"
#include<fstream>

namespace sweeper_ai
{
    int eco_ai_init_interface(void **p, char * config_params)
    {
        EcoEStatus ecoestatus(EStatus_Success);

        //判断配置文件是否存在
        if (NULL == config_params)
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout << "NULL == config_params in eco_ai_init_interface, config_params =" << config_params << std::endl;
            return ecoestatus;
        }

        //   创建业务　类
        EcoTaskInference *ecotaskinference_ = new EcoTaskInference;
        if (NULL == ecotaskinference_ )
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout  <<  "ecotaskinference_ can't memory: NULL == ecotaskinference_" <<  ecoestatus << std::endl;
            eco_ai_deinit_interface((void *)ecotaskinference_);
            return ecoestatus;
        }
        
        // 开辟内存以及相关资源
        ecoestatus = ecotaskinference_->ecoTaskOpen(config_params);
        if (ecoestatus != EStatus_Success)
        {
            std::cout  <<  "ecotaskinference_->ecoTaskOpen() error in eco_ai_init_interface" <<  ecoestatus << std::endl;
            eco_ai_deinit_interface((void *)ecotaskinference_);
            return ecoestatus;
        }
        
        *p = (void *)ecotaskinference_;

        return ecoestatus;
    }

    int eco_ai_run_interface(void *p, const ImageDatas &input_data, EcoInstanceObjectSegs &output_result)
    {
        EcoEStatus ecoestatus(EStatus_Success);
        int model_index(-1);

        model_index = input_data.input_image[0].model_index;

/**********************************************************  输入参数矫验  *******************************************************************************/
        if (NULL == p)
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout  <<  "NULL == p in eco_ai_run_interface" <<std::endl;
            return ecoestatus;
        }

        // 类型转换
        EcoTaskInference *ecotaskinference_ = (EcoTaskInference *)p;

        if (input_data.num_img != output_result.num_image || input_data.num_img <= 0 || output_result.num_image <= 0)
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout  <<  "input_data.num_img and output_result.num_image mast bigger than 0, input_data.num_img != output_result.num_image error in eco_ai_run_interface, input_data.num_img = " <<  input_data.num_img
            << ", output_result.num_image = " << output_result.num_image <<std::endl;
            return ecoestatus;
        }

        if (NULL == output_result.ecoinstaobjseg_)
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout  <<  "output_result.ecoinstaobjseg_ is NULL in eco_ai_run_interface" <<  ecoestatus <<std::endl;
            return ecoestatus;
        }

    if(ecotaskinference_->getAIType() == EM_RGB_AI_TYPE)  // RGB_AI
    {
        for (int imgid = 0; imgid < input_data.num_img; imgid++)
        {
            if (NULL == input_data.input_image[imgid].image_rgb_data_addr )
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout  <<  "input_data.input_image[ " << imgid << " ] is empty in eco_ai_run_interface" <<  ecoestatus 
                      <<  "     input_data.input_image[ " << imgid << " ].image_rgb_data_addr = "   <<  input_data.input_image[imgid].image_rgb_data_addr
                      << std::endl;

                return ecoestatus;
            }
            if(input_data.input_image[imgid].model_id == 2 || input_data.input_image[imgid].model_id == 7)
            {
                if(   input_data.input_image[imgid].image_rgb_height  != 960 || input_data.input_image[imgid].image_rgb_width   != 1280 )
                {
                    ecoestatus = EStatus_InvalidParameter;
                    std::cout  <<  "model_id == 2, input_data.input_image[ " << imgid << " ].image_rgb_height = "   <<  input_data.input_image[imgid].image_rgb_height
                        <<  "     input_data.input_image[ " << imgid << " ].image_rgb_width = "    <<  input_data.input_image[imgid].image_rgb_width << std::endl;
                    return ecoestatus;
                }
            }
            else if(input_data.input_image[imgid].model_id == 21)
            {
                if(   input_data.input_image[imgid].image_rgb_height  != 720 || input_data.input_image[imgid].image_rgb_width   != 1280 )
                {
                    ecoestatus = EStatus_InvalidParameter;
                    std::cout  <<  "model_id == 21, input_data.input_image[ " << imgid << " ].image_rgb_height = "   <<  input_data.input_image[imgid].image_rgb_height
                        <<  "     input_data.input_image[ " << imgid << " ].image_rgb_width = "    <<  input_data.input_image[imgid].image_rgb_width << std::endl;
                    return ecoestatus;
                }
            }
            else
            {
                if(   input_data.input_image[imgid].image_rgb_height  != 384 || input_data.input_image[imgid].image_rgb_width   != 512 )
                {
                    ecoestatus = EStatus_InvalidParameter;
                    std::cout  <<  "input_data.input_image[ " << imgid << " ].image_rgb_height = "   <<  input_data.input_image[imgid].image_rgb_height
                        <<  "     input_data.input_image[ " << imgid << " ].image_rgb_width = "    <<  input_data.input_image[imgid].image_rgb_width << std::endl;
                    return ecoestatus;
                }
            }
        }
    }
    else if(ecotaskinference_->getAIType() == EM_IR_TYPE)   // 红外 AI
    {
        for (int imgid = 0; imgid < input_data.num_img; imgid++)
        {
            if (NULL == input_data.input_image[imgid].image_ir_data_addr )
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout  <<  "input_data.input_image[ " << imgid << " ] is empty in eco_ai_run_interface" <<  ecoestatus 
                      <<  "     input_data.input_image[ " << imgid << " ].image_ir_data_addr = "   <<  input_data.input_image[imgid].image_ir_data_addr
                      << std::endl;

                return ecoestatus;
            }

            if(   input_data.input_image[imgid].image_depth_height  != 480 || input_data.input_image[imgid].image_depth_width   != 640 )
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout  <<  "model_id == 21, input_data.input_image[ " << imgid << " ].image_depth_height = "   <<  input_data.input_image[imgid].image_depth_height
                    <<  "     input_data.input_image[ " << imgid << " ].image_depth_width = "    <<  input_data.input_image[imgid].image_depth_width << std::endl;
                return ecoestatus;
            }
        }
    } 

        //　图片推断     
        ecoestatus = ecotaskinference_->ecoTaskInfer(input_data);
        if (ecoestatus != EStatus_Success)
        {
            std::cout  <<  "ecotaskinference_->ecotaskinfer error in eco_ai_run_interface" <<  ecoestatus <<std::endl;
            return ecoestatus;
        }

        //　结果输出
        for (int i = 0; i < input_data.num_img; i++)
        {
            // 获取第一张图片的输出结果
            output_result.ecoinstaobjseg_[i] = *ecotaskinference_->getCamDectInferResult(model_index);
        }
    // std::cout <<"44444444444444444444444444444" << std::endl;
        return ecoestatus;
    }

    int eco_ai_deinit_interface(void* p)
    {
        int res = 0;
        if (NULL == p)
        {
            std::cout  <<  "p is null in eco_ai_run_interface" <<std::endl;
            return 0;
        }

        EcoTaskInference *ecotaskinference_ = (EcoTaskInference *)p;

        res = ecotaskinference_->ecoTaskClose();
        if (0 != res)
        {
            std::cout << "0 != res in eco_ai_deinit_interface" << std::endl;
            return res;
        }

        if (NULL != ecotaskinference_)
        {
            delete ecotaskinference_;
            ecotaskinference_ = NULL;
        }
        return 0;
    }
}