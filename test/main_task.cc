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




int  splitString(const std::string & strSrc, const std::string& strDelims, std::vector<std::string>& strDest)  
{  
    typedef std::string::size_type ST;  
    std::string delims = strDelims;  
    std::string STR;  
    if(delims.empty()) 
    {
		delims = "/n/r";
    }
  
	ST pos=0, LEN = strSrc.size();  
	while(pos < LEN ){  
		STR="";   
		while( (delims.find(strSrc[pos]) != std::string::npos) && (pos < LEN) ) 
		{
			++pos;  
		}

		if(pos==LEN) 
		{
			return strDest.size();  
		}
		
		while( (delims.find(strSrc[pos]) == std::string::npos) && (pos < LEN) ) 
		{
			STR += strSrc[pos++];
		}
	
        //std::cout << "[" << STR << "]";  
        if( ! STR.empty() )
        {
            strDest.push_back(STR); 
        }  
	}  
	return strDest.size();  
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
    int frame_id( atoi(argv[5]));
    int modle_id( atoi(argv[6]));


    


#ifdef D_DEBUG_SAVETXT

    std::vector<int> model_lists;

    if(frame_id == 0)
    {
        model_lists = {0};
    }
    else
    {
        model_lists = {20,21,22,23};
    }
    std::ofstream ofs;  
    ofs.open("./img/result/00saveresult.txt", std::ios::out );

#endif

    char text[1024];

    EcoAInterface_struct_t *AI_SYMBOL=NULL;
    int img_num = 3;

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
        cv::glob(imgdir,img_paths, true);
    }

    
    EcoAInterfaceCamImg_t *struct_img = new EcoAInterfaceCamImg_t[img_num];
    if (NULL == struct_img)
    {
        std::cerr << "struct_img can't mem in demo_imgs" << std::endl;
    }
    
    
    EcoAInterfaceCtl_t struct_ctr;
    EcoAInterfaceResult_t struct_res;

    memset(&struct_ctr,0,sizeof(struct_ctr));
    memset(&struct_res,0,sizeof(struct_res));
    // std::cout<<"sleep 30s before init..."<<std::endl;
    // std::this_thread::sleep_for(std::chrono::milliseconds(5000));


    // 读取配置参数文件
    std::ifstream t((char *)(config_s.c_str()));
    std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
    std::cout << "aiParam_json read success" << std::endl;

    void * LineDet=AI_SYMBOL->init((char *)str.c_str());
    if(NULL == LineDet)
    {
        return -1;
    }

    // std::cout<<"sleep 30s before run..."<<std::endl;
    // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    // std::cout<<"start run..."<<std::endl;

    int count=1;
    for (int i = 0; i < count; i++){
        
        for(int frameid = 0; frameid < img_paths.size(); ++frameid)
        {

#ifdef D_DEBUG_SAVETXT            
            for(auto iter = model_lists.begin(); iter!= model_lists.end(); iter ++)
            {
                modle_id = *iter;
#endif
                std::string imgs_RGB_path(img_paths[frameid]);

                cv::Mat img_RGB = cv::imread(imgs_RGB_path.c_str(), 1);
                if (img_RGB.empty())
                {
                    fprintf(stderr, "cv::imread %s failed\n", imgs_RGB_path.c_str());
                    return -1;
                }
                std::cout << "imgid = " << frameid << "   image_path = " << imgs_RGB_path.c_str() << std::endl;

                std::string  image_name =imgs_RGB_path.substr(imgs_RGB_path.find_last_of("/") + 1);

                std::string imgs_depth_path = string_replace(imgs_RGB_path, std::string("rgb"), std::string("depth"));
                imgs_depth_path = string_replace(imgs_depth_path, std::string(".jpg"), std::string("_regist.png"));
                // imgs_depth_path = string_replace(imgs_depth_path, std::string(".jpg"), std::string(".png"));
                cv::Mat img_depth = cv::imread(imgs_depth_path.c_str(), -1);
                if (img_depth.empty())
                {
                    fprintf(stderr, "cv::imread %s failed\n", imgs_depth_path.c_str());
                    return -1;
                }
                // std::cout << "depth image read success" << std::endl;

                std::string imgs_ir_path = string_replace(imgs_depth_path, std::string("depth"), std::string("ir"));
                cv::Mat img_ir = cv::imread(imgs_ir_path.c_str(), 0);
                if (img_ir.empty())
                {
                    fprintf(stderr, "cv::imread %s failed\n", imgs_ir_path.c_str());
                    return -1;
                }



                if (1 == frame_id)
                {
                    cv::flip(img_RGB,   img_RGB,   -1);
                    cv::flip(img_depth, img_depth, -1);
                    cv::flip(img_ir,    img_ir,    -1);
                }


                EcoAInterfaceDeebotStatus_t st;                  //　时间与位姿信息，需要外部输入

    /********   读取 RGB 数据 ***********************************************/
                memset(&struct_img[0], 0, sizeof(EcoAInterfaceCamImg_t));
                struct_img[0].img_format = ECOAINTEFACE_IMG_FORMAT_RGB;
                struct_img[0].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(ECOAINTEFACE_IMG_SRC_TYPE_RGBD1 + frame_id);
                struct_img[0].img        = img_RGB.data;
                struct_img[0].img_size   = img_RGB.cols * img_RGB.rows * img_RGB.channels();
                struct_img[0].h          = img_RGB.rows;
                struct_img[0].w          = img_RGB.cols;
                struct_img[0].st         = st;

    /********   读取 depth 数据 ***********************************************/
                memset(&struct_img[1], 0, sizeof(EcoAInterfaceCamImg_t));
                struct_img[1].img_format = ECOAINTEFACE_IMG_FORMAT_DEEP;
                struct_img[1].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(ECOAINTEFACE_IMG_SRC_TYPE_RGBD1 + frame_id);
                struct_img[1].img        = img_depth.data;
                struct_img[1].img_size   = img_depth.cols * img_depth.rows * img_depth.channels();
                struct_img[1].h          = img_depth.rows;
                struct_img[1].w          = img_depth.cols;
                struct_img[1].st         = st;

    /********   读取 ir 数据 ***********************************************/
                memset(&struct_img[2], 0, sizeof(EcoAInterfaceCamImg_t));
                struct_img[2].img_format = ECOAINTEFACE_IMG_FORMAT_INFARED;
                struct_img[2].img_type   = ECOAINTEFACE_IMG_SRC_TYPE_E(ECOAINTEFACE_IMG_SRC_TYPE_RGBD1 + frame_id);
                struct_img[2].img        = img_ir.data;
                struct_img[2].img_size   = img_ir.cols * img_ir.rows * img_ir.channels();
                struct_img[2].h          = img_ir.rows;
                struct_img[2].w          = img_ir.cols;
                struct_img[2].st         = st;
    

    #ifdef D_DEBUG_SAVETXT

                std::string  image_name_ptr =image_name.substr(0, image_name.find_last_of(".") - 1);
                std::vector<std::string>  strDest;
                strDest.clear();
                std::string strDelims = "_";

                splitString(image_name_ptr, strDelims, strDest);

                ofs << 0 << " " << strDest[0] << " " << modle_id << " " <<  strDest[2] << " " << strDest[1] << " " << strDest[3] << " ";

    #endif


                struct_ctr.model_id = modle_id;

                auto start = std::chrono::system_clock::now();

                AI_SYMBOL->run(LineDet, struct_img, img_num, &struct_ctr, &struct_res);


                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "image[ " << i << " ]_process time = " << duration.count() << "ms" << std::endl;

                
                // clock_t   now   =   clock();
                // while(clock() - now < (1000*1000)); 

                //-------------------- 结果解析------------------------

                EcoInstanceObjectSegs* ecoinstanceObjectSegs = (EcoInstanceObjectSegs*)struct_res.res;

                std::cout << "results outimg_cnt:" << ecoinstanceObjectSegs[0].ecoinstaobjseg_[0].outimg_cnt << std::endl;
                std::cout << "results outimg:" << ecoinstanceObjectSegs[0].ecoinstaobjseg_[0].outimg << std::endl;


                for (size_t i = 0; i < ecoinstanceObjectSegs->num_image; i++)
                {

                    EcoInstanceObjectSeg * ecoinstanceobject = &ecoinstanceObjectSegs->ecoinstaobjseg_[i];
            
                    EcoGroundObjectDects *ecogroundobjects = ecoinstanceobject->ecogroundobjects;
                    if (NULL != ecogroundobjects)
                    {

                        for (int outnum = 0; outnum < ecogroundobjects->ngroundobjectnum; outnum++)
                        {
                            if(!ecogroundobjects->ecogroundobject[outnum].bisobjects)
                            {
                                continue;
                            }

                            int label    = ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].inlabel;
                            float conf   = ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].fconfidence;
                            cv::Rect roi = ecogroundobjects->ecogroundobject[outnum].rect;
                
                            // std::cout <<"label = " << label << " prop = " << ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].fconfidence 
                            // << "  x = " << roi.x  << "  y = "<< roi.y << "  isface = " << ecogroundobjects->ecogroundobject[outnum].bisface << std::endl;
                            
                            int outlabel = (int)ecogroundobjects->ecogroundobject[outnum].groundobjectsCls.ptrecogroundobjectscls[0].label;

    #ifdef D_DEBUG_SAVETXT

                            ofs << outlabel << " " << conf << " " << roi.x << " " << roi.y  << " " << roi.x + roi.width << " " << roi.y + roi.height << " ";

    #endif




                            rectangle(img_RGB, roi, vcolor[outlabel%100], 4);

                            if (ecogroundobjects->ecogroundobject[outnum].bisface)
                            {
                                rectangle(img_RGB, ecogroundobjects->ecogroundobject[outnum].face_rect, vcolor[outlabel%100 + 1], 4);
                            }


                            sprintf(text, "%d = %.5f", (int)outlabel, conf);
                            putText(img_RGB, text, cv::Point(roi.x, roi.y + 12),
                                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);

                            if(ecoinstanceobject->maskdata[2 * outnum + 0].bistrue || ecoinstanceobject->maskdata[2 * outnum + 1].bistrue)
                            {
                                cv::Point3f points1 = ecoinstanceobject->maskdata[2 * outnum + 0].keypoint;
                                cv::circle(img_RGB, cv::Point2f(points1.x, points1.y), 10, vcolor[ecoinstanceobject->maskdata[2 * outnum + 0].inlabel% 100 + 1], -1);


                                cv::Point3f points2 = ecoinstanceobject->maskdata[2 * outnum + 1].keypoint;
                                cv::circle(img_RGB, cv::Point2f(points2.x, points2.y), 10, vcolor[ecoinstanceobject->maskdata[2 * outnum + 0].inlabel% 100 + 1], -1);

                                cv::Point2f points3 = ecoinstanceobject->maskdata[2 * outnum + 0].mappos;
                                cv::Point2f points4 = ecoinstanceobject->maskdata[2 * outnum + 1].mappos;

    #ifdef D_DEBUG_SAVETXT


                                ofs << points3.x << " " << points3.y << " " << points4.x << " " << points4.y << " ";

    #endif
                                sprintf(text, "(%d,%d)", (int)points3.x, (int)points3.y);
                                putText(img_RGB, text, cv::Point(roi.x, roi.y + roi.height),
                                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);
                                sprintf(text, "(%d,%d)", (int)points4.x, (int)points4.y);
                                putText(img_RGB, text, cv::Point(roi.x + roi.width, roi.y + roi.height),
                                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);


                            }

                        }

                    }
                    else
                    {
                        if (ecoinstanceobject->maskdata.size() > 0)
                        {
                            for (size_t kk = 0; kk < ecoinstanceobject->maskdata.size(); kk++)
                            {
                                if (ecoinstanceobject->maskdata[kk].bistrue)
                                {
                                    cv::Point3f points = ecoinstanceobject->maskdata[kk].keypoint;
                                    
                                    cv::circle(img_RGB, cv::Point2f(points.x, points.y), 3, vcolor[ecoinstanceobject->maskdata[kk].inlabel% 100 + 2], -1);

    #ifdef D_DEBUG_SAVETXT
                                    ofs << ecoinstanceobject->maskdata[kk].label << " " << ecoinstanceobject->maskdata[kk].fconfidence << " " <<  ecoinstanceobject->maskdata[kk].mappos.x << " " << ecoinstanceobject->maskdata[kk].mappos.y << " ";
    #endif
                                
                                }
                            }
                        }
                    }

    #ifndef D_DEBUG_SAVETXT
                    cv::imwrite("./img/result/" + image_name, img_RGB);
    #endif
                }

                std::cout << "results length:" << struct_res.res_len << std::endl;
                std::cout << "results model_id:" << struct_res.model_id << std::endl;
                std::cout << "results function_name:" << struct_res.function_name << std::endl;


                #ifdef D_DEBUG_SAVETXT

                    ofs << "\n ";

                #endif
            }
#ifdef D_DEBUG_SAVETXT     
        }
#endif
    }
    if(NULL != struct_img)
    {
        delete [] struct_img;
        struct_img =NULL;
    }


#ifdef D_DEBUG_SAVETXT

    ofs.close();

#endif

   

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
