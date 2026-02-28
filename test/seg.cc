
#include"iostream"
#include "algorithm"
#include <sys/stat.h>
#include "eco_ai_seg.h"
#include <chrono>
#include "document.h"
#include <fstream>

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
    int framd_id             = atoi(argv[3]);
    int model_out_id             = atoi(argv[4]);
    cv::String imgs_ir_plate_path;

    // 读取配置参数文件
    std::ifstream t(config_path.c_str());
    std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());

    rapidjson::Document document;
    document.Parse(str.c_str());

    cv::Mat img_RGB = cv::imread(imgs_RGB_path.c_str(), 1);
    if (img_RGB.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
    }
    std::cout << "RGB image read success" << std::endl;


    rapidjson::Value model_params;
    if(document.HasMember("models"))
    {
        model_params = document["models"].GetObject();
    }

    EcoSegInference  ecoseginference;


    for (rapidjson::SizeType npa = 0; npa < model_params.Size(); npa++)
    {

        rapidjson::Value model_sparam = model_params[npa].GetObject();

        if (model_sparam.HasMember("id"))
        {
            int model_id = model_sparam["id"].GetInt();

            if (model_id == model_out_id)
            {
                ecoseginference.ecoSegOpen(model_sparam);
            }
        }
    }

    std::vector<cv::Mat> inputs;

    inputs.push_back(img_RGB);
    int cm_distance(-1);

    if(ecoseginference.getOpenFlag())
    {
        ecoseginference.ecoSegInfer(inputs, cv::Rect(0, 0, inputs[0].cols, inputs[0].rows), cm_distance);

        EcoInstanceObjectSeg * result = ecoseginference.getSegMasks();

        cv::Mat res = result->mask;

        std::vector<EcoKeyPoint> maskdata = result->maskdata;

        for (size_t kk = 0; kk < maskdata.size(); kk++)
        {
            cv::Point3f points = maskdata[kk].keypoint;

            cv::circle(img_RGB, cv::Point2f(points.x, points.y), 10, cv::Scalar(0, 0, 255),-1);
        }

        cv::imwrite("./seg_result.jpg", img_RGB);

    }

    ecoseginference.ecoSegClose();

    return 0;
}