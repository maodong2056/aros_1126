
#include "person_remove.h"
#include <dlfcn.h>
#include <fstream>
#include "inference.h"
#include "unistd.h"

namespace sweeper_ai
{
    EcoAInterface_struct_t *AI_SYMBOL = NULL;
    void * LineDet = NULL;
    EcoAInterfaceCamImg_t *struct_img = NULL;
    void *handle = NULL;

    int person_remove_init()
    {
        std::string lib_so   = "/usr/lib/eco_ai_interface/libecoaisweeper.so";

        std::string config_s = "/etc/conf/ai_params.json";
        if(-1 == access(config_s.c_str(), 0))
        {
            config_s = "/data/ai_params.json";
        }

        // std::string lib_so   = "/data/zhoufeng/rknn_person_remove_Linux/lib/libecoaisweeper.so";
        // std::string config_s = "/data/zhoufeng/rknn_person_remove_Linux/model/ai_params.json";

        char * error;
        // 加载动态链接库
        std::cout << "加载动态链接库:" << lib_so <<std::endl;

        handle = dlopen(lib_so.c_str(),RTLD_LAZY);
        if(!handle)
        {
            return -1;
        }
        dlerror();

        AI_SYMBOL = (EcoAInterface_struct_t *)dlsym(handle, ECOAINTERFACE_SYMBOL_STR);
        if ((error = dlerror()) != NULL) 
        {
            std::cout<<"ECOAINTERFACE_SYMBOL:not funded"<<std::endl;
            return -1;
        }

        // 读取配置参数文件
        std::ifstream t((char *)(config_s.c_str()));
        std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
        std::cout << "aiParam_json read success" << std::endl;

        LineDet = AI_SYMBOL->init((char *)str.c_str());
        std::cout<<"111"<<std::endl;
        if(NULL == LineDet)
        {
            return -1;
        }

        struct_img = new EcoAInterfaceCamImg_t;
        if (NULL == struct_img)
        {
            std::cerr << "struct_img can't mem in demo_imgs" << std::endl;
            return -1;
        }


        return 0;
    }


    int person_remove_run(std::string& image_path)
    {
        EcoAInterfaceCtl_t  struct_ctr;
        EcoAInterfaceResult_t struct_res;

        memset(&struct_ctr,0,sizeof(struct_ctr));
        memset(&struct_res,0,sizeof(struct_res));

        int nperson = -1;


        cv::Mat img_RGB = cv::imread(image_path.c_str(), 1);
        if(img_RGB.empty() || img_RGB.channels() != 3)
        {
            std::cout << "Input image is empty." << std::endl; 
            return 1;
        }
        cv::resize(img_RGB, img_RGB, cv::Size(1280, 960), 0, 0, cv::INTER_LINEAR);

        EcoAInterfaceDeebotStatus_t st;                  //　时间与位姿信息，需要外部输入

    /********   读取 RGB 数据 ***********************************************/
        memset(&struct_img[0], 0, sizeof(EcoAInterfaceCamImg_t));
        struct_img[0].img_format = ECOAINTEFACE_IMG_FORMAT_RGB;
        struct_img[0].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(0);
        struct_img[0].img        = img_RGB.data;
        struct_img[0].img_size   = img_RGB.cols * img_RGB.rows * img_RGB.channels();
        struct_img[0].h          = img_RGB.rows;
        struct_img[0].w          = img_RGB.cols;
        struct_img[0].st         = st;
        struct_img[0].switchON   = 7;
        struct_img[0].timestamp   = 666666666;

        struct_ctr.model_id = 2;
        auto start = std::chrono::system_clock::now();

        AI_SYMBOL->run(LineDet, struct_img, 1, &struct_ctr, &struct_res);

        EcoInstanceObjectSegs* ecoinstanceObjectSegs = (EcoInstanceObjectSegs*)struct_res.res;

        /////不同图像的结果
        for (size_t i = 0; i < ecoinstanceObjectSegs->num_image; i++)
        {
            EcoInstanceObjectSeg * ecoinstanceobject = &ecoinstanceObjectSegs->ecoinstaobjseg_[i];

            EcoGroundObjectDects *ecogroundobjects   = ecoinstanceobject->ecogroundobjects;
            if (NULL != ecogroundobjects)
            {
                /////同一图像中不同目标检测结果
                for (int outnum = 0; outnum < ecogroundobjects->ngroundobjectnum; outnum++)
                {
                    if(ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_PERSON)
                    {
                        nperson = 1;
                    }
                }
            }
        }

        return nperson;
    }


    int person_remove_exit()
    {
        if(NULL != struct_img)
        {
            delete struct_img;
            struct_img =NULL;
        }

        AI_SYMBOL->exit(LineDet);  
        LineDet = NULL;
        dlclose(handle);
        return 0;
    }

}

