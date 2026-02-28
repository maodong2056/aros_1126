#ifndef ECOAIWRAPPER_H_
#define ECOAIWRAPPER_H_
#ifdef __cplusplus
extern "C"{
#endif
/*
* 2022.9.23 by xiang.zhou
*/
#include "EcoAInterface.h"
/*
conf_json
{
	"root":"/",
	"model_dir":"model",
	"hotupdate_dir":"/data/hot-models",
	"core":[5,5],
	"modules":[
		{
			"name":"EcoAiGrass",
			"cameras":["fisheye"]
		},
		{
			"name":"EcoAiScrtHumanDetect",
			"cameras":["overall"]
		}
	]
}
-----------------------------
cams_param_json:
[
{
	"name":"fisheye",
	"crop":{
		"x":0,
		"y":0,
		"w":1920,
		"h":1080
	},
	"inparam":{
		"fx":674.94145524602629,
		"fy":674.69865588835205,
		"cx":625.15471246378763,
		"cy":428.50281293027814,
		"k1":-0.19493768423432117,
		"k2":0.25462233424140546,
		"p1":-0.00060898436274223926,
		"p2":-0.00056661234360314569,
		"k3":0.11,
		"k4":0.22,
		"k5":0.33,
		"k6":0.44,
		"xi":1.5116382152218049
	},
	"outparam":[
		{
			"Rvc00":-0.99875748,
			"Rvc01":0.016787216,
			"Rvc02":0.046921436,
			"Rvc10":-0.016374119,
			"Rvc11":-0.99982381,
			"Rvc12":0.0091745574,
			"Rvc20":0.047067188,
			"Rvc21":0.0083948616,
			"Rvc22":0.99885654,
			"Tvc0":0.3147,
			"Tvc1":0.00,
			"Tvc2":0.32
		}
	],
	"spreeds":["/data/config/charge.bin","/data/config/out.bin"]
},
{
	"name":"overall",
	#...
}
]
*/

int EcoAiWrapperInit(char *conf_json, char *cams_param_json);
int EcoAiWrapperExit(void);

// you should free callback result param
int EcoAiWrapperStart(const char *module_name,EcoAInterfaceCallBack callback,void *userData, const char *pfpsjsonstr);

int EcoAiWrapperStop(const char *module_name);

int EcoAiWrapperSetupThreshold(const char *module_name, int model_id, char *threshold_str);

int EcoAiWrapperSetupThresholds(const char *module_name, char *models_str);

int EcoAiWrapperLoadmap(const char *module_name,char *mapath);

int EcoAiWrapperLoadCamOutParm(const char *module_name,char *cam_name, char *outparam_json_str);

int EcoAiWrapperUpdateModuleFps(const char *module_name, const char *pfpsjsonstr);


int EcoAiWrapperDispatch(EcoAInterfaceCamImg_t **image,int img_cnt);


int EcoAiWrapperUpdateModules();

// you should free return value string
char *EcoAiWrapperInfoDisplay();

#ifdef __cplusplus
}
#endif

#endif
