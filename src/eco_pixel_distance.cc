/*
 * @Description: 
 * @Author: kai.zheng
 * @Date: 2023-03-01 10:20:26
 * @LastEditTime: 2023-06-29 13:29:30
 * @LastEditor: kai.zheng
 */

#ifndef PIXEL2LOCATION_H
#define PIXEL2LOCATION_H

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include "eco_pixel_distance.h"

#define DEBUG false
#define MAX_X 90
#define MIN_X 40
#define MAX_Y 10

namespace sweeper_ai 
{
    // float* ff = new float[table_bufLen];
    float depth_image[2457600];
    float depth_image_ir[1843200];

    using json = nlohmann::json;
    // 公式法测距
    Coordinate_temp pixelTolocWithFormula(float fx, float fy, float cx, float cy, int minrow, int row, int col)
    {
        Coordinate_temp loc;
        Coordinate_temp locZero;
        locZero.x = -1;
        locZero.y = -1;
        if (fx == 0 ||fy == 0)
        {
            //no inner.json 
            return locZero;
        }
        
        float theta1  = atan(0.7 / (structure_h / 100)) + atan((minrow - cy) / fy);
        float theta_x = theta1 - atan((row - cy) / fy);
        float L = structure_h * tan(theta_x);
        if(L < 0)
        {
            //cout<<formula.x<<" "<<formula.y<<endl;
            //cout<<row<<" "<<col<<endl;
            return locZero;
        }
        loc.x = L;
        loc.y = (cx - col) * loc.x / fx;
        if (loc.x < 0)
        {
            //cout<<row<<" "<<col<<endl;
            return locZero;
        }
        return loc;

    }

    Coordinate_temp pixelTolocWithFormula_ir(float fx, float fy, float cx, float cy, int maxrow, float mindist, int row, int col)
    {
        Coordinate_temp loc;
        Coordinate_temp locZero;
        locZero.x = -1;
        locZero.y = -1;
        if (fx == 0 ||fy == 0)
        {
            //no inner.json 
            return locZero;
        }
        
        float theta1  = atan(mindist / 6.48) + atan((maxrow - cy) / fy);
        float theta_x = theta1 - atan((row - cy) / fy);
        float L = structure_h * tan(theta_x);
        if(L < 0)
        {
            //cout<<formula.x<<" "<<formula.y<<endl;
            //cout<<row<<" "<<col<<endl;
            return locZero;
        }
        loc.x = L;
        loc.y = (cx - col) * loc.x / fx;
        if (loc.x < 0)
        {
            //cout<<row<<" "<<col<<endl;
            return locZero;
        }
        return loc;

    }

    static float decode_dis_x(int disx) 
    {
        float dis_x = float(disx) / 255 * 70 + 0;
        return (dis_x);
    }

    static float decode_dis_y(int disy) 
    {
        float dis_y = float(disy) / 255 * 250 - 125;
        return (dis_y);
    }


    int read_json(std::string &filename)
    {
        // cout<<filename<<endl;
        std::ifstream jfile(filename);
	    if(!jfile.is_open())
	    { 
	    	std::cout << "open json file error...!!!!" << std::endl;
            //默认过滤尺寸　0地面　 1家具  2人体 4宠物 
            dict_model[Coordinate(0,1)]   =  Physical_Box_Range(6.0,  25.0,   6.0,  12.0);//width_min width_max height_min height_max
            dict_model[Coordinate(0,4)]   =  Physical_Box_Range(4.0,  45.0,   1.0,  45.0);//15-20??????
            dict_model[Coordinate(0,0)]   =  Physical_Box_Range(10.0, 45.0,   8.0,  50.0);
            dict_model[Coordinate(0,2)]   =  Physical_Box_Range(4.0,  60.0,   0.2,  30.0);
            dict_model[Coordinate(0,3)]   =  Physical_Box_Range(10.0, 1500.0, 0.3,  25.0);//地毯地垫不区分，后续根据概率地图的面积区分
            dict_model[Coordinate(0,5)]   =  Physical_Box_Range(3.5,  75.0,   0.3,  30.0);
            dict_model[Coordinate(0,6)]   =  Physical_Box_Range(0.01, 1500.0, 0.01, 1500.0);//no filter
            dict_model[Coordinate(0,7)]   =  Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(0,8)]   =  Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(0,9)]   =  Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(0,10)]  =  Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(0,11)]  =  Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(0,12)]  =  Physical_Box_Range(-100000, 100000, -100000, 100000);

            dict_model[Coordinate(1,0)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);;//width_min width_max height_min height_max 厘米
            dict_model[Coordinate(1,1)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,2)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,3)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,4)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,5)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,6)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,7)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,8)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,9)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,10)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,11)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,12)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,13)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,14)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,15)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,16)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,17)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,18)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,19)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,20)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,21)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
            dict_model[Coordinate(1,22)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);

            dict_model[Coordinate(2,0)]   = Physical_Box_Range(0, 10000, 0, 10000);    //width_min width_max height_min height_max 厘米
            dict_model[Coordinate(2,1)]   = Physical_Box_Range(0, 10000, 0, 10000);

            dict_model[Coordinate(4,0)]   = Physical_Box_Range(0, 10000, 0, 10000);    //width_min width_max height_min height_max 厘米
            dict_model[Coordinate(4,1)]   = Physical_Box_Range(0, 10000, 0, 10000);

	    }
	    else
        {           //打开成功，解析json数据
            try
            {
                int number;
                int model_number,model_id;
                json root;
                jfile >> root;
                dict_model.clear();
                model_number     = root.at("distance")["model_param"].size();           //at方法是c++ json库中用于从json对象中获取指定名称的值的方法，这里获取模型数量model_number
                std::cout << "model_number: " << model_number << std::endl;
                int seg          = root.at("distance")["segment"].size();
                depth_path       = root.at("distance")["depth_image"];
                structure_h      = root.at("distance")["structure_h"];
                structure_center = root.at("distance")["DIS_TO_CAMERA2CENTER"];
                // cout<<"depth_path"<<depth_path<<" "<<structure_center<<endl;
                for(int z = 0; z < seg; z++)
                {
                    segment.push_back(root.at("distance")["segment"].at(z));
                }
                std::cout << root.at("distance")["segment"].size() << std::endl;
                for(int i = 0; i < model_number; i++)
                {
                    number     = root.at("distance")["model_param"].at(i).at("obj").size();
                    model_id   = root.at("distance")["model_param"].at(i).at("model_id");
                    json root1 = root.at("distance")["model_param"].at(i).at("obj");
                    for(int j = 0; j < number; j++)
                    {
                        dict_model.insert(std::pair<Coordinate,Physical_Box_Range>(Coordinate(model_id,root1.at(j).at("class")),Physical_Box_Range(root1.at(j).at("location").at(0),root1.at(j).at("location").at(1),root1.at(j).at("location").at(2),root1.at(j).at("location").at(3))));
                    }
                }
            }
            catch(...)
            {                   //异常处理
                std::cout<<"json file parse_error...!!!!"<<std::endl;
                //默认过滤尺寸　0地面　 1家具  2人体 4宠物 
                dict_model[Coordinate(0,1)]   =  Physical_Box_Range(6.0,  25.0,   6.0,  12.0);//width_min width_max height_min height_max
                dict_model[Coordinate(0,4)]   =  Physical_Box_Range(4.0,  45.0,   1.0,  45.0);//15-20??????
                dict_model[Coordinate(0,0)]   =  Physical_Box_Range(10.0, 45.0,   8.0,  50.0);
                dict_model[Coordinate(0,2)]   =  Physical_Box_Range(4.0,  60.0,   0.2,  30.0);
                dict_model[Coordinate(0,3)]   =  Physical_Box_Range(10.0, 1500.0, 0.3,  25.0);//地毯地垫不区分，后续根据概率地图的面积区分
                dict_model[Coordinate(0,5)]   =  Physical_Box_Range(3.5,  75.0,   0.3,  30.0);
                dict_model[Coordinate(0,6)]   =  Physical_Box_Range(0.01, 1500.0, 0.01, 1500.0);//no filter
                dict_model[Coordinate(0,7)]   =  Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(0,8)]   =  Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(0,9)]   =  Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(0,10)]  =  Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(0,11)]  =  Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(0,12)]  =  Physical_Box_Range(-100000, 100000, -100000, 100000);

                dict_model[Coordinate(1,0)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);;//width_min width_max height_min height_max 厘米
                dict_model[Coordinate(1,1)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,2)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,3)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,4)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,5)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,6)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,7)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,8)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,9)]   = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,10)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,11)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,12)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,13)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,14)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,15)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,16)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,17)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,18)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,19)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,20)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,21)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);
                dict_model[Coordinate(1,22)]  = Physical_Box_Range(-100000, 100000, -100000, 100000);

                dict_model[Coordinate(2,0)]   = Physical_Box_Range(0, 10000, 0, 10000);//width_min width_max height_min height_max 厘米
                dict_model[Coordinate(2,1)]   = Physical_Box_Range(0, 10000, 0, 10000);
    
                dict_model[Coordinate(4,0)]   = Physical_Box_Range(0, 10000, 0, 10000);//width_min width_max height_min height_max 厘米
                dict_model[Coordinate(4,1)]   = Physical_Box_Range(0, 10000, 0, 10000);
            }
        }   
    }


    int init_depthimage(u8 *table_data, int* maxRowInfo, float fx, float fy, float cx, float cy, int Flag, int& maxdistance)
    {
        // image=cv::imread(filename);
        // // cv::imshow("image",image);
        // // cv::waitKey(0);
        if(Flag)
        {
            std::cout << "distance.dat is open!!!!" << std::endl;
            //如果读到了深度图直接赋值
            std::ifstream ifs(depth_path, std::ios::binary | std::ios::in);
            if (!ifs)
            {
                std::cout << "file is not open!!!!" << std::endl;
                return -1;
            }
            else
            {
                std::cout << "file is open" << std::endl;
	            ifs.read((char*)depth_image, sizeof(float) * table_bufLen);
            }
	        ifs.close();
            return 0;                  //读到了返回0，没读到返回-1
        }
        else
        {
            if (table_data == NULL) 
            {
                std::cout << "table_data is null" << std::endl;
                for(int i = 0; i < sizeof(depth_image) / sizeof(depth_image[0]); i++)
                {
                    depth_image[i] = -1;
                }
            // 指针为空
            }
            else
            {
                std::cout << "distance.dat is not open!!!!" << std::endl;          
                int minrow, mincol, maxrow, maxcol;    
                //截取的深度图像的范围：最小行数为maxRowInfo[0]和maxRowInfo[2]中的较小值，最大行数为maxRowInfo[0]和maxRowInfo[2]中的较大值，最小列数为maxRowInfo[1]和maxRowInfo[3]中的较小值，最大列数为maxRowInfo[1]和maxRowInfo[3]中的较大值
                if(maxRowInfo[2] > maxRowInfo[0])
                {
                    minrow = maxRowInfo[0];
                    maxrow = maxRowInfo[2];
                }
                else
                {
                    minrow = maxRowInfo[2];
                    maxrow = maxRowInfo[0];
                }

                if(maxRowInfo[3] > maxRowInfo[1])
                {
                    mincol = maxRowInfo[1];
                    maxcol = maxRowInfo[3];
                }
                else
                {
                    mincol = maxRowInfo[3];
                    maxcol = maxRowInfo[1];
                }

                for(int row = 0; row < 960; row++)
                {
                    for(int col = 0; col < 1280; col++)
                    {
                    
                        //if(col>=mincol&& col<=maxcol && row>=minrow && row<=maxrow){
                        if(row >= minrow && row <= maxrow && col >= 30 && col < 1250)
                        {
                            //查表法
                            int dis_x = (int) (table_data[2 * ((row - IMG_ROW_RANGE[0]) * table_width +
                                                            (col - IMG_COL_RANGE[0]))]);//计算该像素点对应的loc.x(解码前)
                            int dis_y = (int) (table_data[
                                    2 * ((row - IMG_ROW_RANGE[0]) * table_width + (col - IMG_COL_RANGE[0])) + 1]);

                            if(dis_x == 0)
                            {
                                //查表法表x为零时
                                Coordinate_temp formula = pixelTolocWithFormula(fx, fy, cx, cy, minrow, row, col);
                                // cout<<"minrow"<<minrow<<endl;
                                // exit(0);
                                // fin.write((char *)&formula.x, sizeof(float)); 
                                // fin.write((char *)&formula.y, sizeof(float));
                                image.at<float>(row,col)             = formula.x;
                                depth_image[(row*IMG_WIDTH+col)*2]   = formula.x;
                                depth_image[(row*IMG_WIDTH+col)*2+1] = formula.y + structure_center;
                                if(formula.x < MAX_X && formula.x > MIN_X && maxdistance < 0 && abs(formula.y) < MAX_Y)
                                {
                                    maxdistance = row;
                                    std::cout << "maxdistance = " << maxdistance << "  formula.x = " << formula.x << std::endl;
                                }
                            }
                            else
                            {
                                float code_x_seg = (float) decode_dis_x(dis_x);
                                float code_y_seg = (float) decode_dis_y(dis_y);


                                depth_image[(row*IMG_WIDTH+col)*2]   = code_x_seg;
                                depth_image[(row*IMG_WIDTH+col)*2+1] = code_y_seg - 2.7 + structure_center;
                                image.at<float>(row,col)             = code_x_seg;
                                if(code_x_seg < MAX_X && code_x_seg > MIN_X && maxdistance < 0 && abs(code_y_seg) < MAX_Y)
                                {
                                    maxdistance = row;
                                    std::cout << "maxdistance = " << maxdistance << "  code_x_seg = " << code_x_seg << std::endl;
                                }
                                // fin.write((char *)&code_x_seg, sizeof(float)); 
                                // fin.write((char *)&code_y_seg, sizeof(float)); 
                            }    
                        }
                        else
                        {
                            //公式法
                            Coordinate_temp formula = pixelTolocWithFormula(fx, fy, cx, cy, minrow, row, col);

                            // fin.write((char *)&formula.x, sizeof(float)); 
                            // fin.write((char *)&formula.y, sizeof(float)); 
                            image.at<float>(row,col)             = formula.x;
                            depth_image[(row*IMG_WIDTH+col)*2]   = formula.x;
                            depth_image[(row*IMG_WIDTH+col)*2+1] = formula.y + structure_center;
                            if(formula.x < MAX_X && formula.x > MIN_X && maxdistance < 0 && abs(formula.y) < MAX_Y)
                            {
                                maxdistance = row;
                                std::cout << "maxdistance = " << maxdistance << "  formula.x = " << formula.x << std::endl;
                            }
                        }
                    }
                }
                // imwrite("/data/distance_map.jpg",image);
            }
        }
    }   

    int init_depthimage_ir(u8 *table_data, int* maxRowInfo, float fx, float fy, float cx, float cy, int Flag, int& maxdistance)
    {
        // image=cv::imread(filename);
        // // cv::imshow("image",image);
        // // cv::waitKey(0);
        if(Flag)
        {
            std::cout << "distance.dat is open!!!!" << std::endl;
            //如果读到了深度图直接赋值
            std::ifstream ifs(depth_path, std::ios::binary | std::ios::in);
            if (!ifs)
            {
                std::cout << "file is not open!!!!" << std::endl;
                return -1;
            }
            else
            {
                std::cout << "file is open" << std::endl;
	            ifs.read((char*)depth_image_ir, sizeof(float) * table_bufLen_ir);
            }
	        ifs.close();
            return 0;                  //读到了返回0，没读到返回-1
        }
        else
        {
            if (table_data == NULL) 
            {
                std::cout << "table_data is null" << std::endl;
                for(int i = 0; i < sizeof(depth_image_ir) / sizeof(depth_image_ir[0]); i++)
                {
                    depth_image_ir[i] = -1;
                }
            // 指针为空
            }
            else
            {
                std::cout << "distance.dat is not open!!!!" << std::endl;          

                int min_row_all = INT_MAX;

                for (int col = 0; col < 1280; ++col) {
                    int first_row = maxRowInfo[2 * col];
                    int last_row = maxRowInfo[2 * col + 1];

                    if (first_row != -1 && last_row != -1) {

                        if (first_row < min_row_all) {
                            min_row_all = first_row;
                        }
                    }
                }
                if (min_row_all == INT_MAX) {
                    min_row_all = 400;
                }

                for(int row = 0; row < 720; row++)
                {
                    for(int col = 0; col < 1280; col++)
                    {
                        int minrow = maxRowInfo[2 * col];
                        int maxrow = maxRowInfo[2 * col + 1];
                    
                        //if(col>=mincol&& col<=maxcol && row>=minrow && row<=maxrow){
                        if( col >= 30 && col < 1250          // 列范围
                        && minrow != -1 && maxrow != -1  // 该列有有效非0行
                        && row >= minrow && row <= maxrow )
                        {
                            //查表法
                            int dis_x = (int) (table_data[2 * ((row - IMG_ROW_RANGE_ir[0]) * table_width_ir +
                                                            (col - IMG_COL_RANGE_ir[0]))]);//计算该像素点对应的loc.x(解码前)
                            int dis_y = (int) (table_data[
                                    2 * ((row - IMG_ROW_RANGE_ir[0]) * table_width_ir + (col - IMG_COL_RANGE_ir[0])) + 1]);
                            
                            if(dis_x == 0)
                            {
                                //查表法表x为零时
                                Coordinate_temp formula = pixelTolocWithFormula(fx, fy, cx, cy, min_row_all, row, col);
                                // cout<<"minrow"<<minrow<<endl;
                                // exit(0);
                                // fin.write((char *)&formula.x, sizeof(float)); 
                                // fin.write((char *)&formula.y, sizeof(float));
                                image.at<float>(row,col)             = formula.x;
                                depth_image_ir[(row*IMG_WIDTH_ir+col)*2]   = formula.x;
                                depth_image_ir[(row*IMG_WIDTH_ir+col)*2+1] = formula.y + structure_center;
                                if(formula.x < MAX_X && formula.x > MIN_X && maxdistance < 0 && abs(formula.y) < MAX_Y)
                                {
                                    maxdistance = row;
                                    std::cout << "maxdistance = " << maxdistance << "  formula.x = " << formula.x << std::endl;
                                }
                            }
                            else
                            {
                                float code_x_seg = (float) decode_dis_x(dis_x);
                                float code_y_seg = (float) decode_dis_y(dis_y);

                                depth_image_ir[(row*IMG_WIDTH_ir+col)*2]   = code_x_seg;
                                depth_image_ir[(row*IMG_WIDTH_ir+col)*2+1] = code_y_seg - 2.7 + structure_center;
                                image.at<float>(row,col)             = code_x_seg;
                                if(code_x_seg < MAX_X && code_x_seg > MIN_X && maxdistance < 0 && abs(code_y_seg) < MAX_Y)
                                {
                                    maxdistance = row;
                                    std::cout << "maxdistance = " << maxdistance << "  code_x_seg = " << code_x_seg << std::endl;
                                }
                                // fin.write((char *)&code_x_seg, sizeof(float)); 
                                // fin.write((char *)&code_y_seg, sizeof(float)); 
                            }    
                        }
                        else
                        {
                            Coordinate_temp formula;
                            formula.x = -1.0f;
                            formula.y = -1.0f;

                            if (col >= 30 && col < 1250)
                            {
                                if (minrow == -1 || maxrow == -1) {
                                    formula = pixelTolocWithFormula(fx, fy, cx, cy, min_row_all, row, col);
                                }
                                else{
                                    if (row > maxrow)
                                    {
                                        float mindist = decode_dis_x(table_data[2 * ((maxrow - IMG_ROW_RANGE_ir[0]) * table_width_ir +
                                            (col - IMG_COL_RANGE_ir[0]))]);
                                        formula = pixelTolocWithFormula_ir(fx, fy, cx, cy, maxrow, mindist, row, col);
                                    }
                                    else
                                    {
                                        float maxdist = decode_dis_x(table_data[2 * ((minrow - IMG_ROW_RANGE_ir[0]) * table_width_ir +
                                            (col - IMG_COL_RANGE_ir[0]))]);
                                        formula = pixelTolocWithFormula_ir(fx, fy, cx, cy, minrow, maxdist, row, col);
                                    }
                                }
                            }
                            else if (col < 30)
                            {
                                int target_col = 30;
                                minrow = maxRowInfo[2 * target_col];
                                if(minrow == -1) {
                                    formula = pixelTolocWithFormula(fx, fy, cx, cy, min_row_all, row, col);
                                }
                                else{
                                    float maxdist = decode_dis_x(table_data[2 * ((minrow - IMG_ROW_RANGE_ir[0]) * table_width_ir +
                                        (target_col - IMG_COL_RANGE_ir[0]))]);
                                    formula = pixelTolocWithFormula_ir(fx, fy, cx, cy, minrow, maxdist, row, col);
                                }
                            }
                            else
                            {
                                int target_col = 1250;
                                minrow = maxRowInfo[2 * target_col];
                                if(minrow == -1) {
                                    formula = pixelTolocWithFormula(fx, fy, cx, cy, min_row_all, row, col);
                                }
                                else{
                                    float maxdist = decode_dis_x(table_data[2 * ((minrow - IMG_ROW_RANGE_ir[0]) * table_width_ir +
                                        (target_col - IMG_COL_RANGE_ir[0]))]);
                                    formula = pixelTolocWithFormula_ir(fx, fy, cx, cy, minrow, maxdist, row, col);
                                }
                            }

                            // fin.write((char *)&formula.x, sizeof(float)); 
                            // fin.write((char *)&formula.y, sizeof(float)); 
                            image.at<float>(row,col)             = formula.x;
                            depth_image_ir[(row*IMG_WIDTH_ir+col)*2]   = formula.x;
                            depth_image_ir[(row*IMG_WIDTH_ir+col)*2+1] = formula.y + structure_center;
                            if(formula.x < MAX_X && formula.x > MIN_X && maxdistance < 0 && abs(formula.y) < MAX_Y)
                            {
                                maxdistance = row;
                                std::cout << "maxdistance = " << maxdistance << "  formula.x = " << formula.x << std::endl;
                            }
                        }
                    }
                }
                // imwrite("/data/distance_map.jpg",image);
            }
        }
    }   

    float cal_height(float width, float roi_height, float roi_width) 
    {
        float roi_height_res = -1;
        if(roi_width > 0)
        {
            roi_height_res = width/ roi_width * roi_height;
        }
        return roi_height_res;
    }

    
    bool box_filter(float width,float height,int channel,int label)
    {
        //根据宽高进行过滤
        bool boxFilterJudge = false;
        if (dict_model.count(Coordinate(channel,label)) == 1)
        {
            Physical_Box_Range box_range = dict_model[Coordinate(channel,label)];
            if (width <= box_range.width_max && width >= box_range.width_min && height <= box_range.height_max && height >= box_range.height_min)
            {
                boxFilterJudge = true;
                return boxFilterJudge;
            }
            else{
                boxFilterJudge = false;
                return boxFilterJudge;
            }

        }
        else
        {
            boxFilterJudge = true;
            return boxFilterJudge;
        }
    }


    ///depth_image是全局变量
    point process(int channel, int label, float *a)
    {              //进行目标距离计算
        //接收数据查图中对应的值并进行过滤
        // cout<<"data process start"<<endl;
        float point_x1,point_x2;
        float point_y1,point_y2;
        float width;
        float height = -1;
        point point_return;
   
        point locZero;
        //返回测距结果为-1
        locZero.x1        = -1;
        locZero.x2        = -1;
        locZero.y1        = -1;
        locZero.y2        = -1;
        locZero.objHeight = -1;
        locZero.score     = *(a+4);
        locZero.label     = label;
        locZero.model     = channel;

    if(channel == 100)
    {
        //如果是分割模型,3号模型
        if ((int)*(a) < 0 || (int)*(a + 2) >= IMG_WIDTH || (int)*(a + 1) < 0 || (int)*(a + 3) >= IMG_HEIGHT)
        {
            return locZero;
        }
        point_x1 = depth_image[((int)*(a + 1) * IMG_WIDTH + (int)*(a)) * 2];
        point_y1 = depth_image[((int)*(a + 1) * IMG_WIDTH + (int)*(a)) * 2 + 1];
        point_x2 = depth_image[((int)*(a + 3) * IMG_WIDTH + (int)*(a + 2)) * 2];
        point_y2 = depth_image[((int)*(a + 3) * IMG_WIDTH + (int)*(a + 2)) * 2 + 1];
        if(point_x1 != -1)
        {
            width = point_y1 - point_y2;
            float roi_height = (int)*(a + 3) - (int)*(a + 1);
            float roi_width  = (int)*(a + 2) - (int)*(a);
            height = cal_height(width, roi_height, roi_width);
        }
    }
    else
    {
        //如果是分割模型,3号模型
        if ((int)*(a) < 0 || (int)*(a + 2) >= IMG_WIDTH || (int)*(a + 1) < 0 || (int)*(a + 3) >= IMG_HEIGHT)
        {
            return locZero;
        }
        point_x1 = depth_image[((int)*(a + 3) * IMG_WIDTH + (int)*(a)) * 2];
        point_y1 = depth_image[((int)*(a + 3) * IMG_WIDTH + (int)*(a)) * 2 + 1];
        point_x2 = depth_image[((int)*(a + 3) * IMG_WIDTH + (int)*(a + 2)) * 2];
        point_y2 = depth_image[((int)*(a + 3) * IMG_WIDTH + (int)*(a + 2)) * 2 + 1];
        if(point_x1 != -1)
        {
            width = point_y1 - point_y2;
            float roi_height = (int)*(a + 3) - (int)*(a + 1);
            float roi_width  = (int)*(a + 2) - (int)*(a);
            height = cal_height(width, roi_height, roi_width);
        }
    }

        // bool filter=box_filter(width,height,channel,label);//过滤宽高不符合条件的
        bool filter=true;///临时改的，不过滤任何条件
        if(filter)
        {
            point_return.x1        = point_x1;
            point_return.x2        = point_x2;
            point_return.y1        = point_y1;
            point_return.y2        = point_y2;
            point_return.objHeight = height;
            point_return.score     = *(a+4);
            point_return.label     = label;
            point_return.model     = channel;

            return point_return;
        }
        else
        {    
            return locZero;
        }
    }

    point process_ir(int channel, int label, float *a)
    {              //进行目标距离计算
        //接收数据查图中对应的值并进行过滤
        // cout<<"data process start"<<endl;
        float point_x1,point_x2;
        float point_y1,point_y2;
        float width;
        float height = -1;
        point point_return;
   
        point locZero;
        //返回测距结果为-1
        locZero.x1        = -1;
        locZero.x2        = -1;
        locZero.y1        = -1;
        locZero.y2        = -1;
        locZero.objHeight = -1;
        locZero.score     = *(a+4);
        locZero.label     = label;
        locZero.model     = channel;

        if(channel == 100)
        {
            //如果是分割模型,3号模型
            if ((int)*(a) < 0 || (int)*(a + 2) >= IMG_WIDTH_ir || (int)*(a + 1) < 0 || (int)*(a + 3) >= IMG_HEIGHT_ir)
            {
                return locZero;
            }
            int col_m = ((int)*(a) + (int)*(a + 2)) / 2;
            int row_m = ((int)*(a + 1) + (int)*(a + 3)) / 2;
            point_x1 = depth_image_ir[((int)*(a + 1) * IMG_WIDTH_ir + col_m) * 2];
            point_y1 = depth_image_ir[(row_m * IMG_WIDTH_ir + (int)*(a)) * 2 + 1];
            point_x2 = depth_image_ir[((int)*(a + 3) * IMG_WIDTH_ir + col_m) * 2];
            point_y2 = depth_image_ir[(row_m * IMG_WIDTH_ir + (int)*(a + 2)) * 2 + 1];
            if(point_x1 != -1)
            {
                width = point_y1 - point_y2;
                float roi_height = (int)*(a + 3) - (int)*(a + 1);
                float roi_width  = (int)*(a + 2) - (int)*(a);
                height = cal_height(width, roi_height, roi_width);
            }
        }
        else
        {
            //如果是分割模型,3号模型
            if ((int)*(a) < 0 || (int)*(a + 2) >= IMG_WIDTH_ir || (int)*(a + 1) < 0 || (int)*(a + 3) >= IMG_HEIGHT_ir)
            {
                return locZero;
            }
            point_x1 = depth_image_ir[((int)*(a + 3) * IMG_WIDTH_ir + (int)*(a)) * 2];
            point_y1 = depth_image_ir[((int)*(a + 3) * IMG_WIDTH_ir + (int)*(a)) * 2 + 1];
            point_x2 = depth_image_ir[((int)*(a + 3) * IMG_WIDTH_ir + (int)*(a + 2)) * 2];
            point_y2 = depth_image_ir[((int)*(a + 3) * IMG_WIDTH_ir + (int)*(a + 2)) * 2 + 1];
            if(point_x1 != -1)
            {
                width = point_y1 - point_y2;
                float roi_height = (int)*(a + 3) - (int)*(a + 1);
                float roi_width  = (int)*(a + 2) - (int)*(a);
                height = cal_height(width, roi_height, roi_width);
            }
        }

        // bool filter=box_filter(width,height,channel,label);//过滤宽高不符合条件的
        bool filter=true;///临时改的，不过滤任何条件
        if(filter)
        {
            point_return.x1        = point_x1;
            point_return.x2        = point_x2;
            point_return.y1        = point_y1;
            point_return.y2        = point_y2;
            point_return.objHeight = height;
            point_return.score     = *(a+4);
            point_return.label     = label;
            point_return.model     = channel;

            return point_return;
        }
        else
        {    
            return locZero;
        }
    }

}

#endif