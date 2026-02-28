/*
 * @Description: 
 * @Author: kai.zheng
 * @Date: 2023-03-01 10:42:13
 * @LastEditTime: 2023-04-17 16:58:33
 * @LastEditor: kai.zheng
 */

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <iostream>
#include <algorithm>
#include <vector>

#include <math.h>
#include <sys/types.h>
#include <string>
#include <map>
#define IMG_WIDTH 1280
#define IMG_HEIGHT 960
#define IMG_WIDTH_ir 1280
#define IMG_HEIGHT_ir 720

namespace sweeper_ai 
{
    typedef  unsigned char u8;
    static const int IMG_ROW_RANGE[] = {450, 940};
    static const int IMG_COL_RANGE[] = {30, 1250};
    static int table_bufLen=2*(IMG_WIDTH)*(IMG_HEIGHT);
    const int table_width = IMG_COL_RANGE[1] - IMG_COL_RANGE[0] + 1;

    static const int IMG_ROW_RANGE_ir[] = {360, 720};
    static const int IMG_COL_RANGE_ir[] = {30, 1250};
    static int table_bufLen_ir = 2*(IMG_WIDTH_ir)*(IMG_HEIGHT_ir);
    const int table_width_ir = IMG_COL_RANGE_ir[1] - IMG_COL_RANGE_ir[0] + 1;
    typedef struct Physical_Box_Range_S {
        Physical_Box_Range_S() {}

        Physical_Box_Range_S(float p1, float p2, float p3, float p4) {
            width_min = p1;
            width_max = p2;
            height_min = p3;
            height_max = p4;
        }

        float width_min;
        float width_max;
        float height_min;
        float height_max;

    }Physical_Box_Range;

   //定义物体框真实的物理尺寸
    typedef struct Physical_Box_S{
        float height;//物理宽度
        float width;//物理高度
    }Physical_Box;


    typedef struct Point_S{
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        float objHeight;
        int label;
        int model;
    }point;

    typedef struct Coordinate_S{
        Coordinate_S(){}

        Coordinate_S(int p1, int p2) {
            model = p1;
            label = p2;
        }
        int model;
        int label;
        bool operator <(const Coordinate_S& other_) const
        {
            if (model < other_.model)        //src_port按升序排序
            {
                return true;
            }
            else if (model == other_.model)  //src_port相同，按dst_port升序排序
            {
                if(label < other_.label)
                {
                    return true;
                }


            }
            return false;
        }
    }Coordinate;


    typedef struct Coordinate_temp_S{
    float x;
    float y;
    bool operator <(const Coordinate_temp_S& other_) const
    {
        if (x < other_.x)        //src_port按升序排序
        {
            return true;
        }
        else if (x == other_.x)  //src_port相同，按dst_port升序排序
        {
            if(y < other_.y)
            {
                return true;
            }
        }
        return false;
    }
    }Coordinate_temp;
}
#endif