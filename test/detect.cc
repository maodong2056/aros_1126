
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
    int framd_id             = atoi(argv[3]);
    int model_out_id         = atoi(argv[4]);

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

    cv::Mat img_RGB = cv::imread(imgs_RGB_path.c_str(), 1);
    if (img_RGB.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
    }
    std::cout << "RGB image read success" << std::endl;

    rapidjson::Value modelsparams;
    if (document.HasMember("models"))
    {
       modelsparams = document["models"].GetArray(); 
    }

    EcoDetectInference  ecodetectinference;

    for (rapidjson::SizeType npa = 0; npa < modelsparams.Size(); npa++)
    {

        rapidjson::Value model_sparam = modelsparams[npa].GetObject();

        if (model_sparam.HasMember("id"))
        {
            int model_id = model_sparam["id"].GetInt();

            if (model_id == model_out_id)
            {
                ecodetectinference.ecoDetectOpen(model_sparam);
            }
        }
    }

    std::vector<cv::Mat> inputs;

    inputs.push_back(img_RGB);

    if(ecodetectinference.getOpenFlag())
    {

        ecodetectinference.ecoDetectInfer(inputs, cv::Rect(0, 0, inputs[0].cols, inputs[0].rows));

        EcoGroundObjectDects *result = ecodetectinference.getDetectObjects();

        ecodetectinference.showDetectObjets(inputs[0]);
    }

    ecodetectinference.ecoDetectClose();

    return 0;
}
