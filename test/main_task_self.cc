/*
created by zhou feng,
email:fen.zhou@ecovacs.com,
2022.11.29
*/
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <dlfcn.h>
#include <thread>
#include "eco_ai_defs.h"
#include <unistd.h>
#include <vector>
#include <ctime>
#include <fstream>
#include <vector>
#include <unistd.h>


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



std::vector<cv::Scalar> vcolor={cv::Scalar(125, 0, 255), cv::Scalar(0, 255, 125), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255)
, cv::Scalar(255, 255, 0), cv::Scalar(125, 255, 0), cv::Scalar(255, 0, 125), cv::Scalar(0, 255, 125), cv::Scalar(125, 0, 125), cv::Scalar(0, 225, 0)
, cv::Scalar(255, 255, 0), cv::Scalar(125, 255, 0), cv::Scalar(255, 0, 125), cv::Scalar(0, 255, 125), cv::Scalar(125, 0, 125), cv::Scalar(0, 225, 0)
, cv::Scalar(255, 255, 0), cv::Scalar(125, 255, 0), cv::Scalar(255, 0, 125), cv::Scalar(0, 255, 125), cv::Scalar(125, 0, 125), cv::Scalar(0, 225, 0)
, cv::Scalar(255, 255, 0), cv::Scalar(125, 255, 0), cv::Scalar(255, 0, 125), cv::Scalar(0, 255, 125), cv::Scalar(125, 0, 125), cv::Scalar(0, 225, 0)
, cv::Scalar(255, 255, 0), cv::Scalar(125, 255, 0), cv::Scalar(255, 0, 125), cv::Scalar(0, 255, 125), cv::Scalar(125, 0, 125), cv::Scalar(0, 225, 0)};

int demo_imgs(int argc, char **argv)
{
    std::string lib_so(argv[1]);
    std::string imgdir(argv[2]);
    std::string savedir(argv[3]);
    std::string config_s(argv[4]);
    int frame_id( atoi(argv[5]));////目前frame_id为0是图像未去畸变，frame_id为1是图像去过畸变-ECOAINTEFACE_IMG_SRC_TYPE_E(frame_id)
    int modle_id( atoi(argv[6]));///模型所属id
    int image_id = 0;
    int nres = 0;
    if(argc > 7)
    {
        image_id = atoi(argv[7]);
    }


    char text[1024];

    EcoAInterface_struct_t *AI_SYMBOL = NULL;
    int img_num = 1;        /////这是代表每次读取几张图片？

    char * error;
    // 加载动态链接库
    std::cout << "加载动态链接库:" << lib_so <<std::endl;
    void *handle = NULL;
    handle = dlopen(lib_so.c_str(),RTLD_LAZY);
    if(!handle){
        fprintf(stderr,"%s\n",dlerror());
        exit(EXIT_FAILURE);
    }
    dlerror();

    AI_SYMBOL = (EcoAInterface_struct_t *)dlsym(handle, ECOAINTERFACE_SYMBOL_STR);
    if ((error = dlerror()) != NULL) {
        std::cout<<"ECOAINTERFACE_SYMBOL:not funded"<<std::endl;
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

    /*******   读取 RGB 图片路径********************************/
    std::vector<cv::String> img_paths;
    if (imgdir.find(".jpg") != -1)
    {
        img_paths.push_back(imgdir);
    }
    else
    {
        cv::glob(imgdir, img_paths, true);
    }

    
    EcoAInterfaceCamImg_t *struct_img = new EcoAInterfaceCamImg_t[img_num];
    if (NULL == struct_img)
    {
        std::cerr << "struct_img can't mem in demo_imgs" << std::endl;
    }
    
    
    EcoAInterfaceCtl_t    struct_ctr;
    EcoAInterfaceResult_t struct_res;

    memset(&struct_ctr, 0, sizeof(struct_ctr));
    memset(&struct_res, 0, sizeof(struct_res));
    // std::cout<<"sleep 30s before init..."<<std::endl;
    // std::this_thread::sleep_for(std::chrono::milliseconds(5000));


    // 读取配置参数文件
    std::ifstream t((char *)(config_s.c_str()));
    std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
    std::cout << "aiParam_json read success" << std::endl;

    void * LineDet=AI_SYMBOL->init((char *)str.c_str());
    std::cout<<"111"<<std::endl;
    if(NULL == LineDet)
    {
        return -1;
    }

    int count=1;
    for (int i = 0; i < count; i++)
    {    
        for(int frameid = image_id; frameid < img_paths.size(); ++frameid)
        {
            // sleep(1);
                /////frameid对应的是图片的索引
                std::string imgs_RGB_path(img_paths[frameid]);
                std::string image_name = imgs_RGB_path.substr(imgs_RGB_path.find_last_of("/") + 1);
                std::string timespace = image_name.substr(0, imgs_RGB_path.find_last_of(".") - 1);
                cv::Mat     img_RGB;

                if(modle_id == 21)
                {
                    img_RGB = cv::imread(imgs_RGB_path.c_str(), 0);
                    if (img_RGB.empty())
                    {
                        fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
                        continue;
                    }

                    std::cout << "imgid = " << frameid << "   image_path = " << imgs_RGB_path.c_str() << std::endl;

                    EcoAInterfaceDeebotStatus_t st;                  //　时间与位姿信息，需要外部输入

        /********   读取 RGB 数据 ***********************************************/
                    memset(&struct_img[0], 0, sizeof(EcoAInterfaceCamImg_t));
                    struct_img[0].img_format = ECOAINTEFACE_IMG_FORMAT_INFARED;
                    // struct_img[0].img_format = ECOAINTEFACE_IMG_FORMAT_BGR;
                    // struct_img[0].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(ECOAINTEFACE_IMG_SRC_TYPE_RGBD1 + frame_id);
                    struct_img[0].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(frame_id);
                    struct_img[0].img        = img_RGB.data;
                    struct_img[0].img_size   = img_RGB.cols * img_RGB.rows * img_RGB.channels();
                    struct_img[0].h          = img_RGB.rows;
                    struct_img[0].w          = img_RGB.cols;
                    struct_img[0].st         = st;
                    struct_img[0].switchON   = 7;
                    struct_img[0].timestamp  = std::atof(timespace.c_str());

                    struct_ctr.model_id = modle_id;

                }
                else
                {
                    img_RGB = cv::imread(imgs_RGB_path.c_str(), 1);
                    if (img_RGB.empty())
                    {
                        fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
                        continue;
                    }
                    // cv::cvtColor(img_RGB, img_RGB, cv::COLOR_BGR2RGB);
 
                    if(modle_id == 2)
                    {
                        cv::resize(img_RGB, img_RGB, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
                    }
                    else
                    {
                        if(img_RGB.cols != 512 || img_RGB.rows != 384)
                        {
                            cv::resize(img_RGB, img_RGB, cv::Size(512, 384), 0, 0, cv::INTER_LINEAR);
                        }
                    }
                    cv::resize(img_RGB, img_RGB, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
                    // cv::imwrite("/data/test/dibao_3562_test/512_384.jpg", img_RGB);
                    if (img_RGB.empty())
                    {
                        fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
                        continue;
                    }
                    std::cout << "imgid = " << frameid << "   image_path = " << imgs_RGB_path.c_str() << std::endl;

                    EcoAInterfaceDeebotStatus_t st;                  //　时间与位姿信息，需要外部输入

        /********   读取 RGB 数据 ***********************************************/
                    memset(&struct_img[0], 0, sizeof(EcoAInterfaceCamImg_t));
                    struct_img[0].img_format = ECOAINTEFACE_IMG_FORMAT_RGB;
                    // struct_img[0].img_format = ECOAINTEFACE_IMG_FORMAT_BGR;
                    // struct_img[0].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(ECOAINTEFACE_IMG_SRC_TYPE_RGBD1 + frame_id);
                    struct_img[0].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(frame_id);
                    struct_img[0].img        = img_RGB.data;
                    struct_img[0].img_size   = img_RGB.cols * img_RGB.rows * img_RGB.channels();
                    struct_img[0].h          = img_RGB.rows;
                    struct_img[0].w          = img_RGB.cols;
                    struct_img[0].st         = st;
                    struct_img[0].switchON   = 7;
                    struct_img[0].timestamp  = std::atof(timespace.c_str());

                    struct_ctr.model_id = modle_id;
                }

                auto start = std::chrono::system_clock::now();
                std::cout << "11111111111111111111" << std::endl;
                AI_SYMBOL->run(LineDet, struct_img, img_num, &struct_ctr, &struct_res);
                std::cout << "222222222222222222222" << std::endl;

                //-------------------- 结果解析------------------------

                EcoInstanceObjectSegs* ecoinstanceObjectSegs = (EcoInstanceObjectSegs*)struct_res.res;

                // cv::resize(img_RGB, img_RGB, cv::Size(1280, 960),  0, 0, cv::INTER_LINEAR);
                // cv::resize(img_RGB, img_RGB, cv::Size(1440, 1080), 0, 0, cv::INTER_LINEAR);
                if(modle_id == 2)
                {
                    cv::resize(img_RGB, img_RGB, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
                }
                else if(modle_id == 21)
                {
                    cv::resize(img_RGB, img_RGB, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    cv::resize(img_RGB, img_RGB, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);
                }

                bool flag = false;
                /////不同图像的结果
                for (size_t i = 0; i < ecoinstanceObjectSegs->num_image; i++)
                {

                    EcoInstanceObjectSeg * ecoinstanceobject = &ecoinstanceObjectSegs->ecoinstaobjseg_[i];
            
                    EcoGroundObjectDects *ecogroundobjects = ecoinstanceobject->ecogroundobjects;
                    nres = 0;
                    if (NULL != ecogroundobjects)
                    {
                        /////同一图像中不同目标检测结果
                        // nres = ecogroundobjects->ngroundobjectnum;
                        for (int outnum = 0; outnum < ecogroundobjects->ngroundobjectnum; outnum++)
                        {
                            ////这句是从测距库里面出来时bisobjects为true/false,分别代表其测距是否为-1，来决定是否跳过，临时去掉为了测试
                            // if(!ecogroundobjects->ecogroundobject[outnum].bisobjects)
                            // {
                            //     continue;
                            // }

                            /////下面的inlabel就是从模型得到的cls_id赋值过来的
                            int label    = ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].inlabel;
                            float conf   = ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].fconfidence;
                            ////roi是检测结果框--单个检测框--(roi.x,roi.y)左上角点--roi.width宽和roi.height高
                            cv::Rect roi = ecogroundobjects->ecogroundobject[outnum].rect;

                            int outlabel = (int)ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].label;
                            if(outlabel > 100 && outlabel < 110)
                            {
                                nres++;
                            }


                            if (!ecogroundobjects->ecogroundobject[outnum].bisobjects)
                            {
                                continue;
                            }
                            
                            rectangle(img_RGB, roi, vcolor[outlabel%100], 1);
                            // cv::imwrite("./output_pic/2.jpg", img_RGB);

                            sprintf(text, "%d = %.5f", (int)outlabel, conf);    /////显示的是映射之后的outlabel
                            // sprintf(text, "%d = %.5f", (int)label, conf);    /////显示的是映射之前的label
                            putText(img_RGB, text, cv::Point(roi.x, roi.y - 50),
                                        cv::FONT_HERSHEY_SIMPLEX, 1, vcolor[outlabel%100], 2);
                            if(modle_id == 1)
                            {
                                sprintf(text, "%d = %d", (int)ecogroundobjects->ecogroundobject[outnum].objectprop.direction, (int)ecogroundobjects->ecogroundobject[outnum].objectprop.shape);    /////显示的是映射之后的outlabel
                                // sprintf(text, "%d = %.5f", (int)label, conf);    /////显示的是映射之前的label
                                putText(img_RGB, text, cv::Point(roi.x, roi.y + 50),
                                            cv::FONT_HERSHEY_SIMPLEX, 1, vcolor[outlabel%100], 2);
                            }
                            


                            ////图像上的左下点Image_LeftDownPoint和右下点Image_RightDownPoint
                            cv::Point3f Image_LeftDownPoint(roi.x, roi.y + roi.height, -1);
                            cv::Point3f Image_RightDownPoint(roi.x + roi.width, roi.y + roi.height, -1);
                            if(outlabel >= 101 && outlabel <= 104)
                            {
                                Image_LeftDownPoint.y = roi.y;
                            }
                            cv::circle(img_RGB, cv::Point2f(Image_LeftDownPoint.x, Image_LeftDownPoint.y), 5, vcolor[outlabel% 100 + 1], -1);
                            cv::circle(img_RGB, cv::Point2f(Image_RightDownPoint.x, Image_RightDownPoint.y), 5, vcolor[outlabel% 100 + 1], -1);

                            ////测距后的左点和右点
                            cv::Point3f distance_LeftDownPoint  = ecogroundobjects->ecogroundobject[outnum].position[0];
                            cv::Point3f distance_RightDownPoint = ecogroundobjects->ecogroundobject[outnum].position[1];
                            // std::cout << "label = " << outlabel << "  conf = " <<  conf << std::endl;
                            // std::cout << " x1 = " << distance_LeftDownPoint.x  << "   y1 = " << distance_LeftDownPoint.y  << std::endl;
                            // std::cout << " x2 = " << distance_RightDownPoint.x << "   y2 = " << distance_RightDownPoint.y  << std::endl;

                            sprintf(text, "%.2f  %.2f", distance_LeftDownPoint.x, distance_LeftDownPoint.y);    /////显示的是映射之后的outlabel
                            // sprintf(text, "%d = %.5f", (int)label, conf);    /////显示的是映射之前的label
                            putText(img_RGB, text, cv::Point(roi.x, roi.y + roi.height),
                                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 0.5);

                            sprintf(text, "%.2f  %.2f", distance_RightDownPoint.x, distance_RightDownPoint.y);    /////显示的是映射之后的outlabel
                            // sprintf(text, "%d = %.5f", (int)label, conf);    /////显示的是映射之前的label
                            putText(img_RGB, text, cv::Point(roi.x + roi.width, roi.y + roi.height),
                                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 0.5);
                        }

                    }

                    ////之前是NULL != ecogroundobjects时说明maskdata可能只存了检测的结果，如果ecogroundobjects为空时则说明maskdata中只存了分割的结果
                    ////现在是ecogroundobjects和maskdata各自只存储分割/检测的测距结果，可能某个模型使得两个变量同时有值
                    ////所以现在两者不需要用到if和else，可以同时拥有
                    std::cout << "ecoinstanceobject->maskdata.size() = " << ecoinstanceobject->maskdata.size() << std::endl;
                    if (ecoinstanceobject->maskdata.size() > 0)
                    {
                        for (size_t kk = 0; kk < ecoinstanceobject->maskdata.size(); kk++)
                        {
                            if (ecoinstanceobject->maskdata[kk].bistrue)
                            {
                                ///
                                // cv::Point3f points = ecoinstanceobject->maskdata[kk].keypoint;
                                cv::Point2f points = ecoinstanceobject->maskdata[kk].mappos;
                                cv::circle(img_RGB, cv::Point2f(points.x, points.y), 3, vcolor[ecoinstanceobject->maskdata[kk].inlabel% 100 + 2], -1);
                                if(ecoinstanceobject->maskdata[kk].label == EM_OUT_LINE)
                                {
                                    flag = true;
                                }
                                // std::cout << "label = " << ecoinstanceobject->maskdata[kk].label << std::endl;
                                // std::cout << "x1 = " <<ecoinstanceobject->maskdata[kk].keypoint.x << "   y1 = " << ecoinstanceobject->maskdata[kk].keypoint.y << std::endl;

                            }
                        }
                    }   

                    if(modle_id != 21)
                    {
                        cv::cvtColor(img_RGB, img_RGB, cv::COLOR_RGB2BGR);
                        if(modle_id == 10)
                        {
                            cv::imwrite(savedir + "/000" + image_name, ecoinstanceobject->mask);
                        }
                        
                    }
                    std::cout <<   savedir + "/" + image_name  << std::endl;
                    // if(nres > 0)
                    {
                        cv::imwrite(savedir + "/" + image_name, img_RGB);
                    }
                    

                }
            }
    }
    if(NULL != struct_img)
    {
        delete [] struct_img;
        struct_img =NULL;
    }

   

    AI_SYMBOL->exit(LineDet);  
    dlclose(handle);
    exit(EXIT_FAILURE);

}

int main(int argc, char **argv)
{
    demo_imgs(argc, argv);
    // demo_video(argc, argv);
    return 0;
}
