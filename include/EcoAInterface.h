#ifndef ECOAINTERFACE_H_
#define ECOAINTERFACE_H_
/*
* 2022.9.23 by xiang.zhou
* 2024.3.12 by hengli.ma
* version: 0.0.2
*	changes：
* 		1. EcoAInterfaceCamImg_t添加lds,sl,imu数据,
*		2. 添加 超声波地毯识别结果和功能开关（污渍，宠物便便等）
*		3. ECOAINTEFACE_IMG_SRC_TYPE_E RGBD7=有sl图,RGBD8=没有sl图
*		4. ECOAINTEFACE_IMG_FORMAT_E 添加bgr类型
* 2024.4.22 by hengli.ma
* version: 0.0.3
*	changes：
* 		1. EcoAInterfaceCamImg_t添加图片的dma_fd 以及dma相关的结构体指针信息
* 2025.4.10 by hengli.ma
* version: 0.0.4
*	changes：
*		1. EcoAInterfaceSlData_t 结构光添加idx表示图片方位信息
* 		2. EcoAInterfaceSlpoint_t 添加强度信息
****************************************************************8
* 2025.12.30 by hengli.ma
* version: 0.0.5
*	changes：
*		1. EcoAInterfaceDot_t 
* 		2. EcoAInterfaceSpotAreas_t  //干涸污渍 地面材质和轮廓的信息
*/
#ifdef __cplusplus
extern "C"{
#endif
#include <stdint.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>

#define ECOAINTERFACE_SYMBOL _EcoAInterface
#define ECOAINTERFACE_SYMBOL_STR "_EcoAInterface"
#define LDS_POINT_CNT (400)
#define SL_POINT_CNT (640)
#define IMU_POINT_CNT (4)



typedef enum
{
	ECOAINTEFACE_IMG_SRC_TYPE_NOGDC = 0,
	ECOAINTEFACE_IMG_SRC_TYPE_GDC,
	ECOAINTEFACE_IMG_SRC_TYPE_FISHEYE,
	ECOAINTEFACE_IMG_SRC_TYPE_OVERVIEW,
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD1, // has tof
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD2, // no tof
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD3, // has tof_deep
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD4, // no tof_deep
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD5, // has tof_ir
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD6, // no tof_ir
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD7, //有sl图 未去畸变
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD8, //无sl图
	ECOAINTEFACE_IMG_SRC_TYPE_RGBD9, //sl图 去畸变
}ECOAINTEFACE_IMG_SRC_TYPE_E;

typedef enum
{
	ECOAINTEFACE_CORE_INDEX_0 = 0,
	ECOAINTEFACE_CORE_INDEX_1 = 1,
	ECOAINTEFACE_CORE_INDEX_2 = 2,
	ECOAINTEFACE_CORE_INDEX_3 = 3,
}ECOAINTEFACE_CORE_INDEX_E;

typedef enum
{
	ECOAINTEFACE_DE_STATUS_SPOT_CHARING = 0,
	ECOAINTEFACE_DE_STATUS_SPOT_OUTSIDE
}ECOAINTEFACE_DE_STATUS_E;


typedef void * (*EcoAInterfaceInitFxn)(char * aiParam_json_str);

//dev status
typedef struct
{
	float x;
	float y;
	float z;
	float Qx;
	float Qy;
	float Qz; //means theta
	//float theta;
	int64_t timestamp;
	ECOAINTEFACE_DE_STATUS_E status;
}EcoAInterfaceDeebotStatus_t;

//dtof激光数据格式
typedef struct {
	float x;
	float y;
	float rho;
	float theta;
	float power;
}EcoAInterfaceLdspoint_t;

typedef struct {
	EcoAInterfaceLdspoint_t ldsPoint[LDS_POINT_CNT];
	int64_t timestamp;
	EcoAInterfaceDeebotStatus_t status;
}EcoAInterfaceLdsData_t;

//结构光数据格式
typedef struct {
	float x;
	float y;
	float z;
	float grayscales;
}EcoAInterfaceSlpoint_t;

typedef struct {
	EcoAInterfaceSlpoint_t slPoint[SL_POINT_CNT];
	int64_t timestamp;
	int64_t idx;
	EcoAInterfaceDeebotStatus_t status;
}EcoAInterfaceSlData_t;

//imu的瞬时角速度
typedef struct {
	float gyros[3];
	float accs[3];
}EcoAInterfaceImu_t;
typedef struct {
	EcoAInterfaceImu_t imu[4];
	int64_t timestamp;
}EcoAInterfaceImuData_t;


typedef enum
{
	ECOAINTEFACE_IMG_FORMAT_NV12 = 0,
	ECOAINTEFACE_IMG_FORMAT_RGB,
	ECOAINTEFACE_IMG_FORMAT_DEEP,
	ECOAINTEFACE_IMG_FORMAT_INFARED,
	ECOAINTEFACE_IMG_FORMAT_RELIABLITY,
	ECOAINTEFACE_IMG_FORMAT_OTHER,
	ECOAINTEFACE_IMG_FORMAT_RGBA,
	ECOAINTEFACE_IMG_FORMAT_BGR,
}ECOAINTEFACE_IMG_FORMAT_E;

typedef enum
{
	ECOAINTEFACE_MODEL_PETSHIT = 0, //宠物粪便 1=开 0=关
	ECOAINTEFACE_MODEL_KELIWU,	//颗粒物
	ECOAINTEFACE_MODEL_STAIN, // 污渍
}ECOAINTEFACE_MODEL_SWITCH_E;

//轮廓点
typedef struct {
	float x;
	float y;
}EcoAInterfaceDot_t;

//地面材质 轮廓信息
typedef struct {
	uint8_t 				texture;
	int 					dot_len;
	EcoAInterfaceDot_t		*dot;
}EcoAInterfaceSpotAreas_t;

typedef struct {
	int 			  spotAreas_len;
	EcoAInterfaceSpotAreas_t  spotAreas[];
}EcoAInterfaceAreas_t;


//input img
typedef struct
{
	ECOAINTEFACE_IMG_FORMAT_E img_format;
	ECOAINTEFACE_IMG_SRC_TYPE_E img_type;
	uint8_t *img;
	int dma_fd;
	int img_size;
	int w;
	int h;
	int64_t timestamp;
	double isp_exposure;
	double isp_gain;
	char * threshold_str; // "[1.0,2.1,1,2]"
	EcoAInterfaceDeebotStatus_t st;	//软同步下图像时间戳相近的机器位姿信息
	EcoAInterfaceDeebotStatus_t st_near;
	EcoAInterfaceLdsData_t ldsPointsData; //软同步下获取的dtof数据
	EcoAInterfaceSlData_t slPointsData; //软同步下获取的结构光数据
	EcoAInterfaceImuData_t imudata;//同图像时间戳下的imu数据（包括float32[] gyros三轴上的陀螺仪数据，float32[] 三轴上的加速度accs）
	uint8_t carpetValue; //超声波地毯识别结果 1为有地毯
	uint8_t switchON; //功能开关（污渍，宠物便便等） ECOAINTEFACE_MODEL_SWITCH_E 按位与的结果
	void *userData;  //dma img的结构体
	EcoAInterfaceAreas_t	**spotAreas; // 地面材质 轮廓信息
	pthread_mutex_t	*spotAreas_lock;
}EcoAInterfaceCamImg_t;

//result 
typedef struct
{
	uint8_t *img;
	int outimg_size;
	int w;
	int h;
	ECOAINTEFACE_IMG_FORMAT_E img_type;
}EcoAInterfaceOutImg_t;
typedef struct{
	void *res;
	int res_len;
	uint64_t timestamp;
	char *function_name;
	int model_id;
	EcoAInterfaceDeebotStatus_t st;
	ECOAINTEFACE_CORE_INDEX_E core;
	ECOAINTEFACE_IMG_SRC_TYPE_E img_type;
	EcoAInterfaceOutImg_t *outimg;
	int outimg_cnt;
}EcoAInterfaceResult_t;

// control
typedef struct
{
	ECOAINTEFACE_CORE_INDEX_E core;
	int model_id;
	//EcoAInterfaceDeebotStatus_t st;
	//EcoAInterfaceResult_t result;
}EcoAInterfaceCtl_t;

typedef struct MdsAiDispatchResult_manager
{
	EcoAInterfaceResult_t *result;
	int timeout;
	char filename[128];
	int eventid;
	struct MdsAiDispatchResult_manager *pnext;
}MdsAiDispatchResult_manager_t;

// EcoAInterfaceResult_t *result -> you should free result 
typedef void (*EcoAInterfaceCallBack)(EcoAInterfaceResult_t *result, void *userData);

typedef int (*EcoAInterfaceRunFxn)(void *handle, EcoAInterfaceCamImg_t *img,int img_cnt, \
	EcoAInterfaceCtl_t *ctl,EcoAInterfaceResult_t *result);

typedef int (*EcoAInterfaceExitFxn)(void *handle);

typedef int (*EcoAInterfaceLoadMapFxn)(void *handle, char *mapath);

typedef int (*EcoAInterfaceLoadCamOutParmFxn)(void *handle,char *cam_name, char *outparam_json_str);

typedef int (*EcoAInterfaceLoadRefresh)(void *handle,int model_id, void *data);

typedef struct
{
	const char *author;
	const char *function_name;
	const char *version;
	EcoAInterfaceInitFxn init;
	EcoAInterfaceRunFxn run;
	EcoAInterfaceExitFxn exit;
	EcoAInterfaceLoadMapFxn loadmap;
	EcoAInterfaceLoadCamOutParmFxn loadcamoutparam;
	EcoAInterfaceLoadRefresh refresh;
}EcoAInterface_struct_t;

#define ECOAI_INTERFACE_DECLARE(__author,__function_name,__version, \
								__init, __run,__exit, __loadmap,__loadcamoutparam,__refresh)  \
	EcoAInterface_struct_t ECOAINTERFACE_SYMBOL = { \
		.author = __author,	\
		.function_name = __function_name, \
		.version = __version, \
		.init = __init,	\
		.run = __run, \
		.exit = __exit, \
		.loadmap = __loadmap, \
		.loadcamoutparam = __loadcamoutparam,\
		.refresh = __refresh, \
}

#ifdef __cplusplus
}
#endif
#endif