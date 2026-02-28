
#include <stdio.h>
#include <vector>


#include "algorithm"
#include <sys/stat.h>
#include "eco_ai_detect.h"
#include <chrono>
#include "document.h"
#include <fstream>

using namespace sweeper_ai;

std::vector<std::string> params={"TopCamStaticTargetDetect","TopCamDynamicTargetDetect",
"DownCamStaticTargetDetect", "DownCamWaterTargetDetect"};

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

    EcoEStatus errstatus(EStatus_Success);

    cv::String config_path   = argv[1]; // again we are using the Opencv's embedded "String" class
    cv::String imgs_RGB_path = argv[2];
    // int framd_id             = atoi(argv[3]);
    // int model_id             = atoi(argv[4]);

    /////这里的model_id代表的是在json文件中，该模型对应的第几个模型，
    ////比如原先的json文件中的23号污渍模型就是对应在json文件中的索引为4，根据这个，就会自动去json文件中找到对应的模型名称及其配置，并且去相应的位置去加载模型
    int model_id             = atoi(argv[3]);


    cv::String imgs_depth_plate_path;

    // 读取配置参数文件
    std::ifstream t(config_path.c_str());
    std::string   str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    rapidjson::Document document;
    document.Parse(str.c_str());

    if (document.HasParseError()) {
        std::cout<<"chrisyoung:loading IndoorDet config file error"<<std::endl;
        return -1;
    }

    rapidjson::Value modelsparams;
    if (document.HasMember("models"))
    {
       modelsparams = document["models"].GetArray(); 
    }
    std::cout<<"model_id: "<<model_id<<std::endl;
    rapidjson::Value detect_params = modelsparams[model_id].GetObject();

    EcoDetectInference  ecodetectinference;
    /////这个函数里面主要就是从json文件中将某些数据读取出来的
    ecodetectinference.ecoDetectOpen(detect_params);

    std::vector<cv::String> imgs_RGB_path_list;
    cv::glob(imgs_RGB_path, imgs_RGB_path_list, false); 

    int s_count = 0;
    for (int i = 0; i < imgs_RGB_path_list.size(); i++)
    {   
        std::cout<<i<<"#######################: "<<imgs_RGB_path_list.size()<<std::endl;
        auto start = std::chrono::system_clock::now();
        cv::String imgs_RGB_path = imgs_RGB_path_list[i];
        std::cout<<"666: "<<imgs_RGB_path<<std::endl;

        cv::Mat img_RGB = cv::imread(imgs_RGB_path.c_str(), 1);
        if (img_RGB.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
            continue;
        }
        std::cout << "RGB image read success" << std::endl;
        ////将图片旋转180度--针对商清图片是颠倒的
        cv::flip(img_RGB, img_RGB, -1);

        std::vector<cv::Mat> inputs;
        inputs.push_back(img_RGB);

        auto end2 = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start);
        std::cout << "load picture time = " << duration.count() << "ms" << std::endl;


        std::cout<<"444: "<<inputs.size()<<std::endl;
        int modelSwitch = 1;
        ecodetectinference.ecoDetectInfer(inputs, cv::Rect(0, 0, inputs[0].cols, inputs[0].rows), modelSwitch);


        auto end3 = std::chrono::system_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2);
        std::cout << "time of each infer = "<< duration2.count() << "ms" << std::endl;
        std::cout<<"555: "<<inputs.size()<<std::endl;

        EcoGroundObjectDects *result = ecodetectinference.getDetectObjects();
        std::cout<<"999: "<<inputs.size()<<std::endl;

        // ecodetectinference.showDetectObjets(inputs[0]);
        s_count += 1;
        std::cout<<"^^^^^^^^^^^^^^^^^^: "<<s_count<<std::endl;

    }

    ecodetectinference.ecoDetectClose();

    return 0;
}
