/*
 * @Description: 
 * @Author: kai.zheng
 * @Date: 2023-03-01 10:19:50
 * @LastEditTime: 2023-04-19 14:03:38
 * @LastEditor: kai.zheng
 */
#ifndef __PIXEL2LOCATION_H__
#define __PIXEL2LOCATION_H__
#include "data_type.h"
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <json.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>

namespace sweeper_ai
{
    static cv::Mat image=cv::Mat::zeros(960,1280,CV_32FC1);;
    static std::map<Coordinate, Physical_Box_Range> dict_model;
    static std::vector<int> segment;
    static std::string depth_path="/data/distance.dat";
    static float structure_h=7.47;//离地高度
    static float structure_center=2.7;//相机距离模组中心距离


//extern "C"{

    int read_json(std::string &filename);
    int init_depthimage(u8 *table_data,int* maxRowInfo,float fx,float fy,float cx,float cy,int Flag, int& maxdistance);
    bool box_filter(float width,float height,int channel,int label);
    point process(int channel,int label,float *a);

    int init_depthimage_ir(u8 *table_data,int* maxRowInfo,float fx,float fy,float cx,float cy,int Flag, int& maxdistance);
    point process_ir(int channel,int label,float *a);
//}


}
#endif
