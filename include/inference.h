/*
 * @Descripttion: 商用地宝项目AI部分对RK3588接口初定v0.1 (Linux板子, Android板子通用)
 * @Author: Zhou Feng
 * @Date: 2022-04-12
 * @LastEditors: Zhou Feng
 * @LastEditTime: 2022-04-12
 */

#ifndef ECO_AI_LIB_INFERENCE_H
#define ECO_AI_LIB_INFERENCE_H

#include"eco_ai_defs.h"
#include"eco_common.h"

namespace sweeper_ai
{
	
	#ifdef __cplusplus
	extern "C" 
	{
	#endif

		/**
		 * Init task from model files
		 * @param[in]  p: return of eco_ai_init_interface handle
		 * @param[in]  config_params: task_config task 所需的配置文件
		 * @param[out] return 0 sucess, return -1 failed
		 */
		int eco_ai_init_interface(void **p, char * config_params);


		/**
		 * model run time interface, apply ai funciton, return ai result
		 * @param[in]  p: return of eco_ai_init_interface handle
		 * @param[in]  input_data
		 * @param[in]  output_result
		 * @param[out] return 0 sucess, return -1 failed
		 */
		int eco_ai_run_interface(void *p, const ImageDatas &input_data, EcoInstanceObjectSegs &output_result);


		/**
		 * @description: destructor the eco_ai_init_interface handle
		 * @param[in] {*}
		 */	
		int eco_ai_deinit_interface(void* p);

	#ifdef __cplusplus
	}
	#endif

} // end namespace sweeper_ai

#endif  // ECO_AI_LIB_RK3588_INTERFACE_H
