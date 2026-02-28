#include "EcoAInterface.h"
#include "inference.h"
#include "eco_task_infer.h"
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include "utils.h"

using namespace sweeper_ai;

// 定义函数句柄
typedef struct _EcoAITaskHandle
{
    _EcoAITaskHandle():
    ecotaskinference_(NULL),input_data(NULL),output_result(NULL)
    {
        function_names.clear();
    }
    void *                ecotaskinference_;
    ImageDatas            *input_data;
    EcoInstanceObjectSegs *output_result;
    std::vector<std::vector<std::string>> function_names;   ////两层vector，外层是放上镜头/下镜头，内层是放对应上镜头中的静态/动态目标
    EcoAInterfaceOutImg_t *outimg;
}EcoAITaskHandle;


// 模型编号数组
const std::vector<int>  rgb_model_ids = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 21};

const std::vector<int>  ir_model_ids  = {21};

static std::vector<int> model_ids;
// 模型功能类型
static std::vector<std::string> topcamparams  = {"EcoAiSweeper"};


void* sweeper_init(char* aiParam_json_str)
{
    int ecoestatus(0);
    EcoAITaskHandle *model_handle(NULL);

    //判断配置文件是否存在
    if(NULL == aiParam_json_str)
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout << "NULL == aiParam_json_str in sweeper_init" << aiParam_json_str << std::endl;
        return NULL;
    }

    // 申请 AI 句柄
    model_handle = new EcoAITaskHandle;
    if(NULL == model_handle)
    {
        ecoestatus = EStatus_OutOfMemory;
        std::cout << "model_handle can't memory in sweeper_init, ecoestatus = " << ecoestatus << std::endl;
        return NULL;
    }

    ecoestatus = (int)eco_ai_init_interface(&model_handle->ecotaskinference_, aiParam_json_str);
    if(0 != ecoestatus)
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout << "eco_ai_init_interface is error in sweeper_init, ecoestatus = " << ecoestatus << std::endl;
        return NULL;
    }

    EcoTaskInference *ecotaskinference_ = (EcoTaskInference *)model_handle->ecotaskinference_;

    if(ecotaskinference_->getAIType() == EM_RGB_AI_TYPE)  // 如果是 RGB AI
    {
        model_ids = rgb_model_ids;
    }
    else if (ecotaskinference_->getAIType() == EM_IR_TYPE) // 如果是 IR AI
    {
        model_ids = ir_model_ids;
    }

    // 输入数据内存申请--根据model_ids的数量申请内存
    model_handle->input_data = new ImageDatas[model_ids.size()];
    if(NULL == model_handle->input_data)
    {
        ecoestatus = EStatus_OutOfMemory;
        std::cout << "NULL == input_data in sweeper_init" << std::endl;
        return NULL;
    } 

    // 输出数据内存申请--根据model_ids的数量申请内存
    model_handle->output_result = new EcoInstanceObjectSegs[model_ids.size()];
    if(NULL == model_handle->output_result)
    {
        ecoestatus = EStatus_OutOfMemory;
        std::cout << "NULL == output_result in sweeper_init" << std::endl;
        return NULL;
    } 

    model_handle->outimg = new EcoAInterfaceOutImg_t[model_ids.size()];
    if(NULL == model_handle->outimg)
    {
        ecoestatus = EStatus_OutOfMemory;
        std::cout << "NULL == outimg in sweeper_init" << std::endl;
        return NULL;
    } 

    for(int nmodeid = 0; nmodeid < model_ids.size(); nmodeid++)
    {

        // 申请输入数据内存--对model_ids中的多个model的输入图片个数及其数据进行申请内存，目前设定各个模型输入图片数都是1
        model_handle->input_data[nmodeid].num_img = 1;

        model_handle->input_data[nmodeid].input_image = new ImageData[model_handle->input_data[nmodeid].num_img];
        if(NULL == model_handle->input_data[nmodeid].input_image)
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout << "NULL == input_data[" << nmodeid << "].input_images in sweeper_init" << std::endl;
            return NULL;
        } 

        // 申请输出数据内存
        model_handle->output_result[nmodeid].num_image = model_handle->input_data[nmodeid].num_img;
        model_handle->output_result[nmodeid].ecoinstaobjseg_ = new EcoInstanceObjectSeg[model_handle->output_result[nmodeid].num_image];
        if(NULL == model_handle->output_result[nmodeid].ecoinstaobjseg_)
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout << "NULL == output_result[" << nmodeid << "].ecoinstaobjseg_ in sweeper_init" << std::endl;
            return NULL;
        }    
    }

    model_handle->function_names.push_back(topcamparams);

    std::string binary_dir = "/data/autostart/7/";
    if (access(binary_dir.c_str(), 0) == 0) 
    {
        int status = system("rm -rf /data/autostart/7/*");
        if(status != 0)
        {
            std::cout << "error!!!!!!!!!!!!!" << std::endl;
        }
    }

    // std::string binary_dir_ = "/data/autostart/4/";
    // if (access(binary_dir_.c_str(), 0) == 0) 
    // {
    //     int status = system("rm -rf /data/autostart/4/*");
    //     if(status != 0)
    //     {
    //         std::cout << "error!!!!!!!!!!!!!" << std::endl;
    //     }
    // }

    std::string binary_dir_indoor = "/data/autostart/1/";
    if (access(binary_dir_indoor.c_str(), 0) == 0) 
    {
        int status = system("rm -rf /data/autostart/1/*");
        if(status != 0)
        {
            std::cout << "error!!!!!!!!!!!!!" << std::endl;
        }
    }

    return  (void *)model_handle;

}

int sweeper_run(void *handle, EcoAInterfaceCamImg_t *img, int img_cnt, 
	EcoAInterfaceCtl_t *ctl, EcoAInterfaceResult_t *result)
{

    // 增加系统时间打印
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);
 
    // 打印 系统 格式化的时间
    std::cout << "now_time : " << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "   " << std::endl;
   
    auto start = std::chrono::system_clock::now();
    EcoEStatus ecoestatus(EStatus_Success);
    int ret(0);

    if (NULL == handle)
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "NULL == handle in sweeper_run" <<std::endl;
        return ecoestatus;
    }

    if (NULL == img)
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "NULL == img in sweeper_run" <<std::endl;
        return ecoestatus;
    }

    if (NULL == img[0].img)
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "NULL == img[0].img in sweeper_run" <<std::endl;
        return ecoestatus;
    }

    // 功能句柄
    EcoAITaskHandle *model_handle = (EcoAITaskHandle*) handle;

    int n_model_id = find( model_ids.begin( ), model_ids.end( ), ctl->model_id ) - model_ids.begin( );   // 模型在模型列表中的编号
    if(n_model_id >= model_ids.size())
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "model_id is valid in sweeper_run" <<std::endl;
        return ecoestatus;
    }

    for (size_t i = 0; i < img_cnt; i++)
    {
        /****** *图像数据输入* *********************************************************************/
        // 目前只有rgb数据，后面的 深度图 和 ir 图目前忽略不计
        if (ECOAINTEFACE_IMG_FORMAT_RGB          == img[i].img_format)
        {
            model_handle->input_data[n_model_id].input_image->image_rgb_data_addr = img[i].img;   // RGB 图片数据内存
            model_handle->input_data[n_model_id].input_image->image_rgb_height    = img[i].h;
            model_handle->input_data[n_model_id].input_image->image_rgb_width     = img[i].w;
            // 实验室功能开关--宠物便便/颗粒物/污渍
            model_handle->input_data[n_model_id].input_image->switchON            = img[i].switchON;

        }
        else if (ECOAINTEFACE_IMG_FORMAT_DEEP    == img[i].img_format)
        {
            model_handle->input_data[n_model_id].input_image->image_depth_data_addr = img[i].img; // DEEP 图片数据内存
            model_handle->input_data[n_model_id].input_image->image_depth_height    = img[i].h;
            model_handle->input_data[n_model_id].input_image->image_depth_width     = img[i].w;
        }
        else if (ECOAINTEFACE_IMG_FORMAT_INFARED == img[i].img_format)
        {
            model_handle->input_data[n_model_id].input_image->image_ir_data_addr = img[i].img;    // IR 图片数据内存
            model_handle->input_data[n_model_id].input_image->image_depth_height = img[i].h;
            model_handle->input_data[n_model_id].input_image->image_depth_width  = img[i].w;
        }
        else
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout  <<  "img[" << i << "].img_format is invalid in sweeper_run" <<std::endl;
            return ecoestatus;
        }
    }

    if (img->img_type != ECOAINTEFACE_IMG_SRC_TYPE_NOGDC && img->img_type != ECOAINTEFACE_IMG_SRC_TYPE_GDC && img->img_type != ECOAINTEFACE_IMG_SRC_TYPE_RGBD7 && img->img_type != ECOAINTEFACE_IMG_SRC_TYPE_RGBD9)
    {            
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "img->img_type is invalid in sweeper_run" << std::endl;
        return ecoestatus;
    }


    /**** *镜头编号输入--目前镜头编号为0的代表未去畸变，frame_id为1的代表去畸变* ***********************************************************************/
    model_handle->input_data[n_model_id].input_image->frame_id     = (int)(img->img_type - ECOAINTEFACE_IMG_SRC_TYPE_NOGDC); // 镜头编号 0 未去畸变图像

    /****** *位姿以及时间戳输入* ********************************************************************/    
    model_handle->input_data[n_model_id].input_image->st           = img[0].st;

    // std::cout << "img[0].timestamp = " << img[0].timestamp <<  "    img[0].st.timestamp = " << img[0].st.timestamp << "   x = " << img[0].st.x  << "  y = " << img[0].st.y  <<  "  Qz = " << img[0].st.Qz << std::endl;

    /****** *位姿时间戳修改* ***********************************************************************/    
    model_handle->input_data[n_model_id].input_image->st.timestamp = static_cast<std::int64_t>(getTimeStamp());
    std::cout << "   img[0].timestamp = " << img[0].timestamp <<  "   ctl->model_id = " << ctl->model_id << std::endl;     // 打印时间戳

    /****** *模型ID输出* **************************************************************************/ 
    /////这里的ctl->model_id就是指令输入的最后的modle_id
    model_handle->input_data[n_model_id].input_image->model_id     = ctl->model_id;
    model_handle->input_data[n_model_id].input_image->model_index  = n_model_id;


    if(1 == ctl->model_id)
    {
        model_handle->input_data[n_model_id].input_image->ldsPointsData = img[0].ldsPointsData;
    }
    if(3 == ctl->model_id)
    {
        model_handle->input_data[n_model_id].input_image->slPointsData = img[0].slPointsData;
    }
    if(14 == ctl->model_id)
    {
        model_handle->input_data[n_model_id].input_image->spotAreas = img[0].spotAreas;
        model_handle->input_data[n_model_id].input_image->spotAreas_lock = img[0].spotAreas_lock;
        printf("[debug]000库程序: 锁地址 = %p\n", img[0].spotAreas_lock);
    }

    // 算法结果输出
    ret = eco_ai_run_interface(model_handle->ecotaskinference_, model_handle->input_data[n_model_id], model_handle->output_result[n_model_id]);
    if(ret != 0)
    {            
        ecoestatus = EStatus_InvalidParameter;
        std::cout << "eco_ai_run_interface is error in sweeper_run" << ecoestatus << std::endl;
        return ecoestatus;
    }

    model_handle->output_result[n_model_id].ecoinstaobjseg_[0].outimg_cnt = img_cnt;
    model_handle->output_result[n_model_id].ecoinstaobjseg_[0].outimg     = img;

    // 算法结果输出
    memset(result, 0, sizeof(EcoAInterfaceResult_t));
    result->model_id      = ctl->model_id;
    result->core          = ctl->core;
    result->timestamp     = img->timestamp;
    result->res           = &model_handle->output_result[n_model_id];
    result->res_len       = sizeof(model_handle->output_result[n_model_id]);
    result->function_name = (char *)(model_handle->function_names[0][0]).c_str(); 
    result->st            = model_handle->input_data[n_model_id].input_image->st;

    result->outimg = &model_handle->outimg[n_model_id];

    if ((ctl->model_id == 6 || ctl->model_id == 7 || ctl->model_id == 11 || ctl->model_id == 14) && (model_handle->output_result[n_model_id].ecoinstaobjseg_->ecogroundobjects->ngroundobjectnum > 0))
    {
        result->outimg_cnt          = 1;
        result->outimg->h           = model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.rows;
        result->outimg->w           = model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.cols;
        result->outimg->outimg_size = model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.rows * model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.cols * model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.channels();
        result->outimg->img_type    = ECOAINTEFACE_IMG_FORMAT_RGB;
        result->outimg->img         = (uint8_t*)model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.data;
    }
    else
    {
        result->outimg_cnt          = 0;
        result->outimg->h           = model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.rows;
        result->outimg->w           = model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.cols;
        result->outimg->outimg_size = model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.rows * model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.cols * model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.channels();
        result->outimg->img_type    = ECOAINTEFACE_IMG_FORMAT_RGB;
        result->outimg->img         = (uint8_t*)model_handle->output_result[n_model_id].ecoinstaobjseg_[0].mask.data;
    }

    auto end = std::chrono::system_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "ctl->model_id = " << ctl->model_id << "   run time =" << duration4.count() << "ms,  img[0].timestamp = " << img[0].timestamp << std::endl;

    return 0;
}


int sweeper_exit(void *handle)
{

    if (NULL ==  handle)
    {
        std::cout << "NULL ==  handle in sweeper_exit" << std::endl;
        return 0;
    }

    EcoAITaskHandle *model_handle = (EcoAITaskHandle*) handle;

    if (NULL != model_handle->ecotaskinference_)
    {
        eco_ai_deinit_interface(model_handle->ecotaskinference_);
    }
    
    for(int nmodl_id =0; nmodl_id < model_ids.size(); nmodl_id++)
    {
        if (NULL != model_handle->output_result[nmodl_id].ecoinstaobjseg_)
        {
            delete[] model_handle->output_result[nmodl_id].ecoinstaobjseg_;
            model_handle->output_result[nmodl_id].ecoinstaobjseg_ = NULL;
        }
        
        if (NULL != model_handle->input_data[nmodl_id].input_image)
        {
            delete[] model_handle->input_data[nmodl_id].input_image;
            model_handle->input_data[nmodl_id].input_image = NULL;
        }
    }
    model_handle->function_names.clear();

    if(NULL != model_handle->input_data)
    {
        delete[] model_handle->input_data;
        model_handle->input_data = NULL;
    }

    if(NULL != model_handle->output_result)
    {
        delete[] model_handle->output_result;
        model_handle->output_result = NULL;
    }

    if(NULL != model_handle->outimg)
    {
        delete[] model_handle->outimg;
        model_handle->outimg = NULL;
    }

    if (NULL != model_handle)
    {
        delete model_handle;
        model_handle = NULL;
    }
    
	return 0;
}

ECOAI_INTERFACE_DECLARE("zhoufeng", \
                        "EcoAiSweeper", \
                        "0.0.1", \
                        sweeper_init, \
                        sweeper_run, \
                        sweeper_exit, \
                        NULL, \
                        NULL,
                        NULL);

