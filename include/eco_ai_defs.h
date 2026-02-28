
/******************************************************************************
模块名　　　　： eco_ai_defs
文件名　　　　： eco_ai_defs.h
相关文件　　　： eco_ai_defs.h
文件实现功能　： 公共定义
作者　　　　　： 周峰
版本　　　　　： 1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        走读人    修改内容
2024/01/08    1.0                     周峰       创建
******************************************************************************/

#ifndef __ECO_AI_DEFS_H__
#define __ECO_AI_DEFS_H__

#include <iostream>
#include <deque>
#include <vector>
#include <map>
#include <mutex>
#ifdef D_DEBUG
    #include <chrono>
#endif

// opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "EcoAInterface.h"

#ifdef __cplusplus
extern "C" 
{
#endif

#ifndef MAX_DECTECT_NUM
    #define  MAX_DECTECT_NUM 48
#endif


#ifndef MODEL_NUM
    #define MODEL_NUM 4
#endif

//==========================公共宏定义==========================
#ifndef MIN
    #define MIN(a, b)   ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
    #define MAX(a, b)   ((a) > (b) ? (a) : (b))
#endif
#ifndef ABS
    #define ABS(x) ((x) >= 0 ? (x) : (-(x)))
#endif

#ifndef MAX_MALLOC_NUM
    #define MAX_MALLOC_NUM 640 * 640 * 3
#endif

#ifndef PI
    #define PI 3.1415927410125732
#endif


//==========================公共结构体定义======================
//　类别枚举
typedef enum EM_In_EcoAIInLabels{
    // 避障类
    EM_BACKGROUND = 0,
    EM_OUT_LINE,                    //  1 - 线缆
    EM_OUT_SHOES,                   //  2 - 鞋子
    EM_OUT_WEIGHT_SCALE,            //  3 - 体重秤
    EM_OUT_BASE,                    //  4 - 底座
    EM_OUT_TRASH_CAN,               //  5 - 垃圾桶
    EM_OUT_CLOTH,                   //  6 - 布类
    EM_OUT_GROUND,                  //  7 - 地面可通行区域
    EM_OUT_CARPET,                  //  8 - 地毯,地垫
    EM_OUT_TASSELS,                 //  9 - 流苏
    EM_OUT_UCHAIR_BASE,             //  10 - U形椅底座
    EM_OUT_PET_POOP,                //  11 - 宠物便便
    EM_OUT_UNI_OBJECT,              //  12 - 通用目标
    EM_OUT_CARPET_EDGE,             //  13 - 地毯边界
    EM_OUT_TASSELS_EDGE,            //  14 - 流苏边界
    EM_OUT_CILL,                    //  15 - 门槛，基石
    EM_OUT_GROUND_TRACK,            //  16 - 地轨
    EM_OUT_CHAIR_LEGS,              //  17 - 桌椅腿   
    EM_OUT_LINT,                    //  18 - 毛絮


    // 清洁类
    EM_OUT_PM = 100,                //  100 - 颗粒物
    EM_OUT_PMS,                     //  101 - 堆颗粒物
    EM_OUT_STAIN,                   //  102 - 污渍
    EM_OUT_MIXEDSTATE,              //  103 - 混合态
    EM_OUT_WATERSTAIN,              //  104 - 水渍
    EM_OUT_LIQUID,                  //  105 - 液体
    EM_OUT_DRIED_STAIN,             //  106 - 干涸污渍

    
    // 家具显示类
    EM_OUT_DOOR_FRAME = 200,        //　200- 门框
    EM_OUT_SOFA,                    //　201- 沙发
    EM_OUT_DINING_TABLES_CHAIRS,    //　202- 餐桌椅
    EM_OUT_TEA_TABLE,               //　203- 茶几
    EM_OUT_TV_CAB,                  //　204- 电视柜
    EM_OUT_BED,                     //　205- 床
    EM_OUT_BEDSIDE_TABLE,           //　206- 床头柜
    EM_OUT_CLOSETOOL,               //　207- 马桶
    EM_OUT_MEAL_SIDE_CAB,           //　208- 餐边柜
    EM_OUT_SHOE_CAB,                //　209- 鞋柜
    EM_OUT_TABLE,                   //　210- 桌子
    EM_OUT_AIR_CONDITION,           //　211- 立式空调
    EM_OUT_WASHER,                  //　212- 洗衣机
    EM_OUT_TV,                      //　213- 电视
    EM_OUT_FRIDGE,                  //　214- 冰箱
    EM_OUT_CUPBOARD,                //　215- 橱柜
    EM_OUT_WARDROBE,                //　216- 衣柜
    EM_OUT_CABINET,                 //　217- 普通柜子
    EM_OUT_BOOKCASE,                //　218- 书柜
    EM_OUT_FLOOR_MIRROR,            //　219- 落地镜
    EM_OUT_ISLAND_PLATFORM,         //　220- 岛台
    EM_OUT_GREEN_PLANTS,            //　221- 绿植
    EM_OUT_PIANO,                   //　222- 钢琴
    EM_OUT_GRATE,                   //　223- 壁炉
    EM_OUT_OVEN,                    //　224- 烤箱
    EM_OUT_DISH_WASHER,             //　225- 洗碗机

    // 宠物周边显示类
    EM_OUT_DIET,                    //　226- 饮食盆
    EM_OUT_LITTER_BOX,              //　227- 猫砂盆
    EM_OUT_SHACK,                   //　228- 窝盆

    // 地面材质
    EM_OUT_FLOOR_TILE,              //　229- 地砖
    EM_OUT_FLOOR,                   //　230- 地板

    EM_OUT_MID_AIR,                 //　231- 空
    EM_OUT_BORN,                    //　232- 地  

    EM_OUT_TATAMI,                  //　233- 榻榻米
    EM_OUT_URINEPAD,                //　234- 宠物尿垫

    EM_OUT_PURE_TILE,               //  235- 纯色瓷砖

    // 动态目标
    EM_OUT_PERSON = 300,            //　300- 人形 
    EM_OUT_CAT,                     //　301- 猫 
    EM_OUT_DOG,                     //　302- 狗    

    // 床底和非床底
    EM_OUT_UNDERBED = 400,               //  400- 床底
    EM_OUT_NOUNDERBED,                   //  401- 非床底  

    // 其他
    EM_OTHERLABELS = 999
     
}EcoAILabels;


typedef enum EM_In_EcoDirections{
    EM_OUT_FRONT = 0,              // 前方
    EM_OUT_AFTER,                  // 后方
    EM_OUT_LEFT,                   // 左方
    EM_OUT_RIGHT,                  // 右方
    EM_OUT_LEFT_ADN_FRONT,         // 左前
    EM_OUT_RIGHT_ADN_FRONT,        // 右前
    EM_OUT_LEFT_AND_AFTER,         // 左后
    EM_OUT_RIGHT_AND_AFTER,        // 右后

    EM_OTHERDIRECTIONS = 999
}EcoDirections;


typedef enum EM_In_EcoShapes{
    EM_OUT_CIRCLE = 0,              // 圆形
    EM_OUT_RECT,                    // 方形
    EM_OUT_TRIANGLE,                // 三角形
    EM_OUT_AO_SHAPE,                // 凹形
    EM_OUT_ELLOPSE,                 // 椭圆
    EM_OUT_L_SHAPE,                 // L形

    EM_OTHERSHAPES = 999
}EcoShapes;

typedef struct EcoObjectProps_S{
    EcoObjectProps_S():
    inlabel(-1),fconfidence(-1),direction(EM_OTHERDIRECTIONS),shape(EM_OTHERSHAPES)
    {}

    int inlabel;
    float fconfidence;  
    EcoDirections direction;
    EcoShapes shape;

}EcoObjectProps;

// 目标分类
typedef struct EcoGroundObjectCls_S{

    EcoGroundObjectCls_S():
    label(EM_BACKGROUND), fconfidence(-1), inlabel(-1)
    {}

    int inlabel;                                      //　模型输出标签编号
    EcoAILabels label;                                //　分类类型
    float fconfidence;                                //　分类置信度

}EcoGroundObjectCls;

// 多分类结果
typedef struct EcoGroundObjectsCls_S{

    EcoGroundObjectsCls_S():
    ptrecogroundobjectscls(NULL), nobjectsclsnum(-1)
    {}

    int model_id;                                     //  模型ＩＤ 
    int nobjectsclsnum;                               //　分类　topK 或者　多标签分类
    EcoGroundObjectCls* ptrecogroundobjectscls;       //  分类指针

}EcoGroundObjectsCls;

// 特征提取
typedef struct EcoExtractBlob_S{

    EcoExtractBlob_S():
    nbloblen(-1), blob(NULL)
    {}

    int nbloblen;                                     //　特征节点长度
    float* blob;                                      //  特征节点内存，需要外部申请

}EcoExtractBlob;

//　多个特征节点
typedef struct EcoExtractBlobs_S{

    EcoExtractBlobs_S():
    nblobsnum(-1), ptrecoextractblob(NULL)
    {}

    int nblobsnum;                                    //　特征节点个数
    EcoExtractBlob* ptrecoextractblob;                //　节点特征指针，需要外部申请
    int  model_id;                                    //  模型ＩＤ 

}EcoExtractBlobs;


//关键点检测
typedef struct EcoKeyPoint_S{

    EcoKeyPoint_S():
    inlabel(-1), label(EM_BACKGROUND), keypoint(cv::Point3f(-1, -1, -1)), 
    mappos(cv::Point2f(-1,-1)), fconfidence(-1), bistrue(true)
    {}

    bool bistrue;                                    //  关键点是否有效--经过过滤时该测距是否有效，无效时就置为false，有效就置为true
    int  inlabel;                                    //  模型输出类标标签
    EcoAILabels label;                               //  关键点对外输出标签
    float fconfidence;                               //  关键点置信度
    cv::Point3f keypoint;                            //  目标分割结果相对于镜头前的测距坐标--目前应该只放分割结果的测距结果
    cv::Point2f mappos;                              //  关键点　x,ｙ--目前主要放分割算法的点坐标结果
    float id;                                        //  每个点所属于的曲线的id编号
    float total_num_curve;                           //  单张图片上总的分割曲线数量

}EcoKeyPoint;

//　多个关键点
typedef struct EcoKeyPoints_S{
    
    EcoKeyPoints_S():
    nkeypointnum(-1),ptrecokeypoint(NULL)
    {}

    int nkeypointnum;                                //  关键点个数
    EcoKeyPoint* ptrecokeypoint;                     //  关键点指针

}EcoKeyPoints;


//　目标检测
typedef struct EcoGroundObjectDect_S{

    EcoGroundObjectDect_S():
    rect(cv::Rect()), face_rect(cv::Rect()), groundobjectsCls(EcoGroundObjectsCls()),
    bisface(false), ftimeTick(-1), bisheypoint(false), keypoints(EcoKeyPoints()),
    bisextractblob(false), extractblobs(EcoExtractBlobs()), bisobjects(false),
    objectprop(EcoObjectProps())
    {}

    bool bisobjects;                                 //　判断输出目标是否有效
    cv::Rect rect;                                   //　检测目标所在位置--检测框的坐标点(rect.x,rect.y,rect.width,rect.height)
    EcoGroundObjectsCls  groundobjectsCls;           //　检测目标的类别
    EcoObjectProps  objectprop;                      //　目标属性

    bool bisface;                                    //  判断是否检测到人头部（未实现）
    cv::Rect face_rect;                              //  头部 ROI 框（未实现）

    bool bisheypoint;                                //　检测目标是否存在关键点（未实现）
    EcoKeyPoints keypoints;                          //　检测目标关键点列表（未实现）

    bool bisextractblob;                             //　检测目标是否特征提取（未实现）
    EcoExtractBlobs extractblobs;                    //　检测目标提取特征（未实现）

    float ftimeTick;                                 //  检测目标时间戳（未实现）
    std::vector<cv::Point3f> position;               //  存储检测框测距后的的结果(左下点和右下点的测距后的坐标)
    cv::Point2f mapPosition;                         //  检测目标在ｍａｐ的位置(未实现)

    std::vector<float> lds_position;                   ///存储LDS结合AI的测距结果(一排点，每四个一组，前两个是图像坐标系下点坐标，后两个是基于机器的测距结果)

    std::string OCRstring;                           //  OCR 文字识别

}EcoGroundObjectDect;


typedef struct EcoGroundObjectDects_S{

    EcoGroundObjectDects_S():
    ngroundobjectnum(-1),ecogroundobject(NULL)
    {}

    int model_id;                                    //  模型ＩＤ 
    int ngroundobjectnum;                            //　目标检测个数
    EcoGroundObjectDect* ecogroundobject;            //　目标检测指针

}EcoGroundObjectDects;


// 实例分割
typedef struct EcoInstanceObjectSeg_S{

    EcoInstanceObjectSeg_S():
    ecogroundobjects(NULL),frame_id(-1),model_id(-1),outimg_cnt(-1),outimg(NULL)
    {
        maskdata.clear();
    }

    int  frame_id;                                   //  摄像头ＩＤ
    int  model_id;                                   //  模型ＩＤ

	EcoAInterfaceDeebotStatus_t st;                  //　位姿信息
    EcoGroundObjectDects *ecogroundobjects;          //　单帧的全部目标检测结果--加上存储检测后测距的结果
    std::vector<EcoKeyPoint> maskdata;               //  空间关键点的位置--目前改为了只存分割点的坐标和测距结果
    cv::Mat               mask;                      //  单帧分割结果
    EcoGroundObjectCls    imagecls;                  //  单帧分类结果

	int                   outimg_cnt;                // 输入图片个数
	EcoAInterfaceCamImg_t *outimg;                   // 输入图片同步输出

}EcoInstanceObjectSeg;


// 实例分割----多帧图像输出的所有结果
typedef struct EcoInstanceObjectSegs_S{

    EcoInstanceObjectSegs_S():
    num_image(-1),ecoinstaobjseg_(NULL)
    {}

    int num_image;                                   //　图像帧数
    EcoInstanceObjectSeg* ecoinstaobjseg_;           //　多帧图像结果输出 

}EcoInstanceObjectSegs;


//　目标跟踪
typedef struct EcoTrackedObject_S{

    EcoTrackedObject_S():
    groundObjectDect(EcoGroundObjectDect()),objectIndex(-1)
    {}
    
    using SingleDetectedObject = std::deque<EcoGroundObjectDect>;

    int objectIndex;
    EcoGroundObjectDect groundObjectDect;

}EcoTrackedObject;


//==========================错误码定义==========================
typedef enum EM_EcoEStatus{

    EStatus_Success            = 0,                  // 成功
    EStatus_InvalidParameter   = 0x80000001,         // 参数错误
    EStatus_OutOfMemory        = 0x80000002,         // 内存分配失败
    EStatus_InsufficientBuffer = 0x80000003,         // buffer大小不够
    EStatus_GenericError       = 0x80000004,         // 特定错误
    EStatus_Undefined          = 0x80000005          // 未知错误
}EcoEStatus;



#ifdef __cplusplus
}
#endif

#endif // __ECO_AI_DEFS_H__
