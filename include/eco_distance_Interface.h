/*
 * @Description: 
 * @Author: kai.zheng
 * @Date: 2022-08-25 16:35:07
 * @LastEditTime: 2023-04-17 13:38:45
 * @LastEditor: kai.zheng
 */


#ifndef TEST_INFERFACE_H
#define TEST_INFERFACE_H
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <vector>
#include <sys/types.h>

namespace sweeper_ai
{
typedef  unsigned char u8;
typedef struct obj_detect{
        float row1;
        float col1;
        float row2;
        float col2;
        int cl;
        float score;
    }obj_detect;

typedef struct location{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float objHeight;
    int cl;
    std::vector<float> bbox_lds_data;
}location;
//初始化
void ecoDistanceInit(u8 *table_data,int* maxRowInfo,float fx,float fy,float cx,float cy,int Flag, int& maxdistance);
// void ecoInitinterface();
//数据传入的接口

location ecoDistanceCalProcess(obj_detect roi, std::vector<std::vector<float>>robot_lds_data, std::vector<int> imu, int channel);

void ecoDistanceInit_ir(u8 *table_data,int* maxRowInfo,float fx,float fy,float cx,float cy,int Flag, int& maxdistance);

location ecoDistanceCalProcess_ir(obj_detect roi, std::vector<std::vector<float>>robot_lds_data, std::vector<int> imu, int channel);

//DistanceCal
}
#endif //TEST_INFERFACE_H
