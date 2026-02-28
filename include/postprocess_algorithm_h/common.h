#ifndef _RKNN_MODEL_ZOO_COMMON_H_
#define _RKNN_MODEL_ZOO_COMMON_H_
#include "rknn_api.h"
#include "eco_common.h"


#define OBJ_NAME_MAX_SIZE 64
// #define OBJ_NUMB_MAX_SIZE 128
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM_INDOOR_YOLOV7 36
#define OBJ_CLASS_NUM_INDOOR_YOLOV8 17
#define DIR_CLASS_NUM_INDOOR_YOLOV8 4
#define OBJ_CLASS_NUM_MULTITASK 8
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE_INDOOR_YOLOV7 (5 + OBJ_CLASS_NUM_INDOOR_YOLOV7)
#define OBJ_CLASS_NUM_PM 4 //80
#define OBJ_CLASS_NUM_LIQUID 5 //80
#define OBJ_CLASS_NUM_LINT 1
#define OBJ_CLASS_NUM_DRYSTAIN 5
#define OBJ_CLASS_NUM_IRSTAIN 4
#define PROP_BOX_SIZE_PM (5 + OBJ_CLASS_NUM_PM)
#define PROP_BOX_SIZE_LIQUID (5 + OBJ_CLASS_NUM_LIQUID)
#define PROP_BOX_SIZE_LINT (5 + OBJ_CLASS_NUM_LINT)
#define PROP_BOX_SIZE_DRYSTAIN (5 + OBJ_CLASS_NUM_DRYSTAIN)
#define OBJ_CLASS_NUM_OBSTACLE 4
#define OBJ_CLASS_NUM_PET 2


/**
 * @brief Image rectangle
 * 
 */
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;


typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct {
    float scale_w;
    float scale_h;
} scale_t;


typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;


typedef struct BBox 
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float label;
    float score;
} BBox;

/// 多任务//
typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct _BOX_PROP
{
    int name;
    float condidence;
} BOX_PROP;

typedef struct _detect_result_t
{

    BOX_RECT box;
    std::vector<BOX_PROP> prop;     ////一个检测框对应的多个阈值，一个阈值对应一个类别和该类别的置信度
    std::vector<BOX_PROP> sub_prop;     ////附属标签
    bool issure;

} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];

} detect_result_group_t;

typedef struct{
    int (*func)(sweeper_ai::EcoRknnModelParams& , rknn_output *, letterbox_t *, std::vector<float>, std::vector<float>, float, std::vector<detect_result_t>&,  int, int);
    float  threshold_a;
    float  threshold_b;
    float  threshold_c;
    size_t output_struct_size;
    int    output_struct_number;
    const  char* head_name;
    void*  head_results;
}postprocess_struct;



#endif //_RKNN_MODEL_ZOO_COMMON_H_




