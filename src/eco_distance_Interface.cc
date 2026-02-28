/*
 * @Description: 
 * @Author: kai.zheng
 * @Date: 2022-08-25 16:35:08
 * @LastEditTime: 2023-06-29 13:49:41
 * @LastEditor: kai.zheng
 */
#include "eco_distance_Interface.h"
#include "eco_pixel_distance.h"
namespace sweeper_ai
{
    using namespace std;
    void ecoDistanceInit(u8 *table_data, int* maxRowInfo, float fx, float fy, float cx, float cy, int Flag, int& maxdistance)
    {
        string json_path = "/model/EcoAiSweeper/models.json";
        read_json(json_path);
        // fx = fx / 1.5;
        // fy = fy / 1.5;
        // cx = cx - 320;
        // cy = cy - 60;
        init_depthimage(table_data, maxRowInfo, fx, fy, cx, cy, Flag, maxdistance);
    }

    void ecoDistanceInit_ir(u8 *table_data, int* maxRowInfo, float fx, float fy, float cx, float cy, int Flag, int& maxdistance)
    {
        init_depthimage_ir(table_data, maxRowInfo, fx, fy, cx, cy, Flag, maxdistance);
    }

    location ecoDistanceCalProcess(obj_detect roi, std::vector<std::vector<float>> robot_lds_data, std::vector<int> imu, int channel)
    {   

        float a[5];
        a[0]  = roi.col1;
        a[1]  = roi.row1;
        a[2]  = roi.col2;
        a[3]  = roi.row2;
        a[4]  = roi.score;

        point point_retrun = process(channel, roi.cl, a); 


        location loc;
        loc.cl        = point_retrun.label;
        loc.objHeight = point_retrun.objHeight;
        loc.score     = point_retrun.score;
        loc.x1        = point_retrun.x1;
        loc.x2        = point_retrun.x2;
        loc.y1        = point_retrun.y1;
        loc.y2        = point_retrun.y2;

        if(robot_lds_data.size() > 0)
        {
            for(auto it = robot_lds_data.begin(); it !=  robot_lds_data.end(); it++)
            {
                for(auto it2 = it->begin(); it2 !=  it->end(); it2++)
                {
                    loc.bbox_lds_data.push_back(*it2);
                }
            }
        }

    
        return loc;
    }

    location ecoDistanceCalProcess_ir(obj_detect roi, std::vector<std::vector<float>> robot_lds_data, std::vector<int> imu, int channel)
    {   

        float a[5];
        a[0]  = roi.col1;
        a[1]  = roi.row1;
        a[2]  = roi.col2;
        a[3]  = roi.row2;
        a[4]  = roi.score;

        point point_retrun = process_ir(channel, roi.cl, a); 


        location loc;
        loc.cl        = point_retrun.label;
        loc.objHeight = point_retrun.objHeight;
        loc.score     = point_retrun.score;
        loc.x1        = point_retrun.x1;
        loc.x2        = point_retrun.x2;
        loc.y1        = point_retrun.y1;
        loc.y2        = point_retrun.y2;

        if(robot_lds_data.size() > 0)
        {
            for(auto it = robot_lds_data.begin(); it !=  robot_lds_data.end(); it++)
            {
                for(auto it2 = it->begin(); it2 !=  it->end(); it2++)
                {
                    loc.bbox_lds_data.push_back(*it2);
                }
            }
        }

    
        return loc;
    }
}