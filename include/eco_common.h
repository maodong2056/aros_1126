/******************************************************************************
模块名　　　　： eco_common
文件名　　　　： eco_common.h
相关文件　　　： eco_common.h
文件实现功能　： 用到的通用数据结构体
作者　　　　　： 周峰
版本　　　　　： 1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        创建人    修改内容
2024/01/08    1.0                     周峰         创建
******************************************************************************/


#ifndef __ECO_COMMON_H__
#define __ECO_COMMON_H__

#include"eco_ai_defs.h"
#include"rknn_api.h"


namespace sweeper_ai
{
	
    // resize 类型
    typedef enum EM_RESIZE_TYPE_S{
        // 静态目标
        NONE_RESIZE = -1,
        EM_NORMAL_RESIZE,         // 直接 resize
        EM_PAD_RESIZE,            // 下方pad + resize
        EM_PAD_RESIZE_TOP_BOTTOM, // 上下pad + resize
        EM_RGA_NORMAL_RESIZE,     // RGA 直接resize
        EM_RGA_PAD_RESIZE,        // RGA + pad + resize

        // 其他
        EM_OTHERRESIZE = 999
        
    }EcoResizeTypeS;

    // AI type 枚举类型
    typedef enum EM_AI_TYPE_S{
        // 静态目标
        NONE_TYPE = -1,
        EM_RGB_AI_TYPE,         // RGB_AI
        EM_IR_TYPE,             // IR_AI

        // 其他
        EM_OTHER_TYPE = 999
        
    }EM_AI_TYPE;

    // resize 类型
    typedef enum EM_MODEL_TYPE_S{
        // 静态目标
        NONE_MODEL = -1,
        EM_MODEL_LINE,             // 0 - 高精度电线
        EM_MODEL_OBSTACLE,         // 1 - 障碍物
        EM_MODEL_FREESPACE,        // 2 - freespace
        EM_MODEL_PET_POOP,         // 3 - 宠物便便

        EM_MODEL_PM,               // 4 - 颗粒物 
        EM_MODEL_STAIN,            // 5 - 污渍

        EM_MODEL_PFURNITURE,       // 6 - 家具模型
        EM_MODEL_GROUND_MATERIAL,  // 7 - 地面材质
        EM_MODEL_GROUND_OR_AIR,    // 8 - 悬空属性
        EM_MODEL_RANGIND,          // 9 - 测距模型

        EM_MODEL_PERSON,           // 10 - 人形模型
        EM_MODEL_HUMAN_KEYPOINTS,  // 11 - 人体关键点
        EM_MODEL_2DTO3D,           // 12 - 指哪扫哪
        EM_MODEL_PET,              // 13 - 宠物模型

        // 其他
        EM_OTHERMODEL = 999
        
    }EcoModelTypeS;



    // 相机内参矩阵
    typedef struct CamerainnerInfo_S{

        CamerainnerInfo_S():
        fx(-1),fy(-1),cx(-1),cy(-1),k1(-1),k2(-1),k3(-1),k4(-1),k5(-1),k6(-1),p1(-1),p2(-1)
        {}

        float fx;
        float fy;
        float cx;
        float cy;
        float k1;
        float k2;
        float k3;
        float k4;
        float k5;
        float k6;
        float p1;
        float p2;

    }CamerainnerInfo;


    // 相机参数矩阵
    typedef struct CameraInfo_S{

        CameraInfo_S():
        RGB(CamerainnerInfo()),Depth(CamerainnerInfo()),IR(CamerainnerInfo()),
        Height(-1),rangle(-1),Length(-1),ir_Height(-1),ir_rangle(-1)
        {}

        CamerainnerInfo RGB;
        CamerainnerInfo Depth;
        CamerainnerInfo IR;

        float Height;     // 镜头的物理高度
        float rangle;     // 镜头中每个行对应与光心的角度
        float Length;     // 镜头在地面投影的长度
        float MAX_Rows;   // 镜头在特定距离内的最大行数
        float MIN_Rows;   // 镜头在特定距离内的最小行数
        float MAX_Length; // 镜头识别的最远距离
        float ir_Height;     // 镜头的物理高度
        float ir_rangle;     // 镜头中每个行对应与光心的角度

    }CameraInfo;


    // 输入数据格式
    typedef struct ImageData_S{

        ImageData_S():
        frame_id(-1), image_rgb_data_addr(NULL), image_depth_data_addr(NULL), image_ir_data_addr(NULL),
        image_rgb_height(-1), image_rgb_width(-1), image_depth_height(-1), image_depth_width(-1),
        model_id(-1), imudata_ago(-1), ultrasonic(-1)
        {
            for (size_t i = 0; i < MODEL_NUM; i++)
            {
                model_infer_id[i] = false;
            }
        }

        int  frame_id;                   // 摄像头ＩＤ
        int  model_id;                   // 模型ID
        int  image_rgb_height;           // RGB图像高度
        int  image_rgb_width;            // RGB图像宽度
        uchar* image_rgb_data_addr;      // RGB图像内存

        int  image_depth_height;         // 深度图像高度
        int  image_depth_width;          // 深度图像宽度
        uchar* image_depth_data_addr;    // 深度图像内存
        uchar* image_ir_data_addr;       // 红外图像内存

        EcoAInterfaceDeebotStatus_t st;          //　位姿信息
        int imudata_ago;                         //  瞬时IMU数据
        int ultrasonic;                          //  超声波数据

        bool model_infer_id[MODEL_NUM];          // 模型开关
        int model_index;                         // 模型在模型编号中的 下列
        std::vector<std::vector<float>> points;  // 点云数据


        EcoAInterfaceLdsData_t ldsPointsData;  //软同步下获取的dtof数据
        EcoAInterfaceSlData_t  slPointsData;   //软同步下获取的结构光数据
        EcoAInterfaceImuData_t imudata;        //同图像时间戳下的imu数据（包括float32[] gyros三轴上的陀螺仪数据，float32[] 三轴上的加速度accs）
        uint8_t carpetValue;                   //超声波地毯识别结果 1为有地毯
        uint8_t switchON;                      //功能开关（污渍，宠物便便等） ECOAINTEFACE_MODEL_SWITCH_E 按位与的结果

        EcoAInterfaceAreas_t	**spotAreas;
        pthread_mutex_t	*spotAreas_lock;
        
    }ImageData;


    typedef struct ImageDatas_S{

        ImageDatas_S():
        input_image(NULL), num_img(-1)
        {}
        int num_img;                            // 模型个数--图片个数？？
        ImageData* input_image;                 // 单个模型数据--单个图片数据？？
        
    }ImageDatas;



    // 模型参数结构体
    typedef struct EcoModelParam_S{

        EcoModelParam_S():
        ctx(0), nmodelinputchannel_(NULL), nmodelinputweith_(NULL),
        nmodelinputheight_(NULL),nmodeloutputchannel_(NULL),nmodeloutputweith_(NULL),
        nmodeloutputheight_(NULL),model_data(NULL)
        {
            out_scales.clear();
            out_zps.clear();
            io_num.n_input = -1;
            io_num.n_output = -1;
        }

            unsigned char *model_data;                          // 模型文件
            
            rknn_context ctx;                                   // 模型句柄 
            std::vector<float>    out_scales;                   // 量化后标准
            std::vector<int32_t>  out_zps;                      // 量化后标准

            rknn_input_output_num io_num;                       // 输入输出节点个数

            rknn_tensor_mem* input_mems[1];                     // 零拷贝输入；默认一个输入
            rknn_tensor_mem* output_mems[15];                   // 零拷贝输出；根据模型输出头数量设定

            int * nmodelinputchannel_;                          // 网络输入　宽度
            int * nmodelinputweith_;                            // 网络输入　宽度
            int * nmodelinputheight_;                           // 网络输入　高度
            
            int * nmodeloutputchannel_;                         // 网络输出　通道数
            int * nmodeloutputweith_;                           // 网络输出　宽度
            int * nmodeloutputheight_;                          // 网络输出　高度
    
    }EcoRknnModelParams;









}

#endif


