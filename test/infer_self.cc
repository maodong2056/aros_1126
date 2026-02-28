
#include "inference.h"
#include <fstream>
#include <chrono>
#include "document.h"
#include "iostream"
#include "algorithm"
#include <unistd.h>

#include <stdio.h>
#include <vector>
#include <sys/stat.h>


using namespace sweeper_ai;

std::string string_replace(std::string strBig, const std::string &strsrc, const std::string &strdst)
{
    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();

    while ((pos = strBig.find(strsrc, pos)) != std::string::npos)
    {
        strBig.replace(pos, srclen, strdst);
        pos += dstlen;
    }
    return strBig;
}


int main(int argc, char** argv)
{

    cv::String config_path     = argv[1]; // again we are using the Opencv's embedded "String" class
    cv::String imgs_RGB_path   = argv[2];
    int frame_id               = atoi(argv[3]);   // 测试单个模型的id号
    int model_infer_id         = atoi(argv[4]);   // 测试单个模型的id号

    void * handle = NULL;
    int nerror = 0;

    //判断配置文件是否存在
    if(-1 == access(config_path.c_str(), 0))
    {
        std::cerr << "aiParam_json_str file can't access " << config_path << std::endl;
        return 0;
    }



    cv::Mat img_RGB = cv::imread(imgs_RGB_path.c_str(), 1);
    if (img_RGB.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
        return -1;
    }
    std::cout << "RGB image read success" << std::endl;

    if (1 == frame_id)
    {
        cv::flip(img_RGB, img_RGB, -1);
    }

    // 读取配置参数文件
    std::ifstream t(config_path.c_str());
    std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
    std::cout << "config_path read success" << std::endl;

    nerror = eco_ai_init_interface(&handle, (char *)str.c_str());
    if (nerror !=0 || NULL == handle)
    {
        std::cerr << "eco_ai_init_interface is error, nerror =" << nerror << std::endl;
        return 1;
    }

    ImageDatas input_data;
    input_data.num_img = 1;
    input_data.input_image = new ImageData;

    input_data.input_image->frame_id = frame_id;

    input_data.input_image->image_rgb_data_addr   = img_RGB.data;
    input_data.input_image->image_rgb_height      =img_RGB.rows;
    input_data.input_image->image_rgb_width       = img_RGB.cols;

    input_data.input_image->model_infer_id[model_infer_id % 20] = true;

    EcoInstanceObjectSegs ecoinstanceObjectSegs;
    ecoinstanceObjectSegs.num_image = 1;
    ecoinstanceObjectSegs.ecoinstaobjseg_ = new EcoInstanceObjectSeg;


    auto start = std::chrono::system_clock::now();


    nerror = eco_ai_run_interface(handle, input_data, ecoinstanceObjectSegs);
    if (nerror != 0)
    {
        std::cerr << "eco_ai_init_interface is error, nerror =" << nerror << std::endl;
        return 1;
    }

    auto end2 = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start);
    std::cout << "eco_ai_init_interface time = " << duration.count() << "ms" << std::endl;


    for (size_t i = 0; i < ecoinstanceObjectSegs.num_image; i++)
    {
        std::cout << "save seg result" << std::endl;
        EcoInstanceObjectSeg * ecoinstanceobject = &ecoinstanceObjectSegs.ecoinstaobjseg_[i];
        cv::imwrite("./seg_image.jpg",ecoinstanceobject->mask * 120);


        EcoGroundObjectDects *ecogroundobjects = ecoinstanceobject->ecogroundobjects;
        if (NULL != ecogroundobjects)
        {
            std::cout << "save dectect result" << std::endl;
            for (int outnum = 0; outnum < ecogroundobjects->ngroundobjectnum; outnum++)
            {
                rectangle(img_RGB,ecogroundobjects->ecogroundobject[outnum].rect, cv::Scalar(255, 0, 0), 4);
            }
            cv::imwrite("./detect_image.jpg", img_RGB);
        }
    }
    

    if (NULL != ecoinstanceObjectSegs.ecoinstaobjseg_)
    {
        delete ecoinstanceObjectSegs.ecoinstaobjseg_;
        ecoinstanceObjectSegs.ecoinstaobjseg_ = NULL;
    }

    if (NULL != input_data.input_image)
    {
        delete input_data.input_image;
        input_data.input_image = NULL;
    }

    nerror = eco_ai_deinit_interface(handle);
    if (nerror !=0 )
    {
        std::cerr << "eco_ai_init_interface is error, nerror =" << nerror << std::endl;
        return 1;
    }

    return 0;

}