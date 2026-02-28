#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <sys/time.h>
#include <vector>
#include <deque>
#include <set>
#include <algorithm> 
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include "postprocess.h"
#include "eco_distance_Interface.h"


const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

namespace sweeper_ai
{

    static EcoAILabels inlalbel2outlabel(int model_infer_id, int &inlabel);
    static void inlalbel2outobjprops(EcoObjectProps& objectprop);

    // std::vector<postprocess_struct> POSTPROCESSLIST = {
    //     {yolov8_post_process, 0.5, 0.5, 8., sizeof(object_detect_result_list), 6, "ground_old8",     nullptr},
    //     {yolov8_post_process, 0.5, 0.5, 1., sizeof(object_detect_result_list), 6, "ground_petsshit", nullptr},
    //     {yolov8_post_process, 0.5, 0.5, 3., sizeof(object_detect_result_list), 6, "ground_new3",     nullptr},
    //     {line_post_process, 0.45, 0.7, 360.,sizeof(object_detect_result_list), 2, "ground_line",     nullptr}
    // };

    std::vector<postprocess_struct> POSTPROCESSLIST = {
        {yolov8_post_process, 0.2,  0.5, 6.,  sizeof(object_detect_result_list), 6, "ground_obstacle", nullptr},
        {line_post_process,   0.45, 0.7, 360.,sizeof(object_detect_result_list), 2, "ground_line",     nullptr}
    };


    //　截取　val　输出范围在　［min，max］之间
    inline static int clamp(float val, int min, int max)
    {
        return val > min ? (val < max ? val : max) : min;
    }

    //　计算两个检测框的重合　IOU
    static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1)
    {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    // sigmoid
    static float sigmoid(float x)
    {
        return 1.0 / (1.0 + expf(-x));
    }

    static float unsigmoid(float y)
    {
        return -1.0 * logf((1.0 / y) - 1.0);
    }

    // 截取最大最小值
    inline static int __clip(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    //  F32 数据转量化后数据
    static int qnt_f32_to_affine(float f32, int zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        int res = (int)__clip(dst_val, -128, 127);
        return res;
    }

    // 量化后数据转 F32
    static float deqnt_affine_to_f32(int qnt, int zp, float scale)
    {
        return ((float)qnt - (float)zp) * scale;
    }

    // 比较检测目标，按着最大置信度从大到小进行排序;
    static bool campara_detect_result(detect_result_t& a, detect_result_t& b)
    {
        return a.prop[0].condidence > b.prop[0].condidence;
    }

    // 比较单个检测目标，按着置信度从大到小进行排序;
    static bool campara_detect_prop(BOX_PROP& a, BOX_PROP& b)
    {
        return a.condidence > b.condidence;
    }

    int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
            int filterId, float threshold)
    {
        for (int i = 0; i < validCount; ++i)
        {
            if (order[i] == -1 || classIds[i] != filterId)
            {
                continue;
            }
            int n = order[i];
            for (int j = i + 1; j < validCount; ++j)
            {
                int m = order[j];
                if (m == -1 || classIds[i] != filterId)
                {
                    continue;
                }
                float xmin0 = outputLocations[n * 4 + 0];
                float ymin0 = outputLocations[n * 4 + 1];
                float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
                float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

                float xmin1 = outputLocations[m * 4 + 0];
                float ymin1 = outputLocations[m * 4 + 1];
                float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
                float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

                float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold)
                {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }




    static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                    std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                    float threshold, int32_t zp, float scale,const int num_class)
    {
        int validCount = 0;
        int grid_len = grid_h * grid_w;
        int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
        for (int a = 0; a < 3; a++)
        {
            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    int8_t box_confidence = input[((num_class + 5) * a + 4) * grid_len + i * grid_w + j];
                    if (box_confidence >= thres_i8)
                    {
                        int offset = ((num_class + 5) * a) * grid_len + i * grid_w + j;
                        int8_t *in_ptr = input + offset;
                        float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                        float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                        float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                        float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                        box_x = (box_x + j)   * (float)stride;
                        box_y = (box_y + i)   * (float)stride;
                        box_w = box_w * box_w * (float)anchor[a * 2];
                        box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        int8_t maxClassProbs = in_ptr[5 * grid_len];
                        int maxClassId = 0;
                        for (int k = 1; k < num_class; ++k)
                        {
                            int8_t prob = in_ptr[(5 + k) * grid_len];
                            if (prob > maxClassProbs)
                            {
                                maxClassId = k;
                                maxClassProbs = prob;
                            }
                        }
                        if (maxClassProbs > thres_i8)
                        {
                            objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                            classId.push_back(maxClassId);
                            validCount++;
                            boxes.push_back(box_x);
                            boxes.push_back(box_y);
                            boxes.push_back(box_w);
                            boxes.push_back(box_h);
                        }
                    }
                }
            }
        }
        return validCount;
    }



    // 获取　分类　topk　结果 
    int rknn_cls_GetTop(int background, float *pfProb, float *pfMaxProb, int *pMaxClass, int outputCount, int topNum)
    {
        int i(-1);
        int j(-1);

        #define MAX_TOP_NUM 10
        if (topNum > MAX_TOP_NUM) return 0;

        memset(pfMaxProb, 0, sizeof(float) * topNum);
        memset(pMaxClass, 0xff, sizeof(float) * topNum);

        for (j = 0; j < topNum; j++)
        {
            for (i=0; i<outputCount; i++)
            {
                if ((i == *(pMaxClass+0)) || (i == *(pMaxClass+1)) || (i == *(pMaxClass+2)) ||
                    (i == *(pMaxClass+3)) || (i == *(pMaxClass+4)))
                {
                    continue;
                }

                if (pfProb[i] > *(pfMaxProb+j))
                {
                    *(pfMaxProb+j) = pfProb[i];
                    *(pMaxClass+j) = i;
                }
            }
        }
        return 1;
    }


    // 人脸绑定到对应行人
    void face2person(EcoInstanceObjectSeg *ecotopcamoutputresult_)
    {
        int k = 0;
        cv::Rect body_roi;
        for (size_t i = 0; i < ecotopcamoutputresult_->ecogroundobjects->ngroundobjectnum; i++)
        {
            EcoGroundObjectDect *ecogroundobjects = &ecotopcamoutputresult_->ecogroundobjects->ecogroundobject[i];

            if (ecogroundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label ==  EM_OUT_PERSON)
            {
                for (size_t j = 0; j < ecotopcamoutputresult_->ecogroundobjects->ngroundobjectnum; j++)
                {
                    EcoGroundObjectDect *faceobjects = &ecotopcamoutputresult_->ecogroundobjects->ecogroundobject[j];
                }
                if (!ecogroundobjects->bisface)
                {
                    if (ecogroundobjects->rect.y < 25 && (ecogroundobjects->rect.y + ecogroundobjects->rect.height) > 1050)
                    {
                        continue;
                    }
                    
                    cv::Rect face_roi;
                    face_roi.x = ecogroundobjects->rect.x;
                    face_roi.y = ecogroundobjects->rect.y;
                    face_roi.width = ecogroundobjects->rect.width;
                    face_roi.height = MAX(face_roi.width / 3, ecogroundobjects->rect.height / 7);

                    ecogroundobjects->bisface = true;
                    ecogroundobjects->face_rect = face_roi;
                }

            }
        }

    }

    // 自己目标检测标签的标签映射 与测距离（x，y，z）
    void topcamstataicdectect(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu)
    {

        if(NULL == ecoinstanceobjectseg)
        {
            std::cout << "NULL == ecoinstanceobjectseg in topcamstataicdectect" << std::endl;     
        }

        if(ecoinstanceobjectseg->ecogroundobjects == NULL)
        {
            std::cout << "ecoinstanceobjectseg->ecogroundobjects == NULL in topcamstataicdectect" << std::endl;
        }


        ///////outecogroundobjects代表单帧全部目标检测结果---可能有多个目标检测结果
        EcoGroundObjectDects* outecogroundobjects = ecoinstanceobjectseg->ecogroundobjects;


        /////outecogroundobjects->ngroundobjectnum代表的是目标检测结果的个数
        for (size_t i = 0; i < outecogroundobjects->ngroundobjectnum; i++)
        {

            EcoGroundObjectDect*  groundobjects = &outecogroundobjects->ecogroundobject[i];

#ifdef D_DEBUG
            std:: cout << "prop = " << groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].fconfidence
            << "  label = " << groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].inlabel
            << "  xx = " << groundobjects->rect.x << "   yy = " << groundobjects->rect.y << std::endl;
#endif

            for (size_t j = 0; j < groundobjects->groundobjectsCls.nobjectsclsnum; j++)
            {
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[j].label = inlalbel2outlabel(model_infer_id, groundobjects->groundobjectsCls.ptrecogroundobjectscls[j].inlabel);
            }

            
            inlalbel2outobjprops(groundobjects->objectprop);
            // std::cout << "label = " << groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label
            // << "  inlabel prop = "  << groundobjects->objectprop.inlabel << std::endl;
            obj_detect obj;
            location loc;

            if(model_infer_id == 2 || model_infer_id == 7)
            {
                std::vector<cv::Point2f> src , dst;
                src.resize(2);
                dst.resize(2);
                ////这里src中的值可以做一个图片大小限制
                src[0].x  = groundobjects->rect.x;
                src[0].y  = groundobjects->rect.y;
                src[1].x  = groundobjects->rect.x + groundobjects->rect.width;
                src[1].y  = groundobjects->rect.y + groundobjects->rect.height;
                std::cout << "src[0].x = " << src[0].x << "  src[1].x = " << src[1].x << std::endl;

                cv::Mat K(3, 3, CV_64FC1, cv::Scalar(0));
                cv::Mat D(8, 1, CV_64FC1, cv::Scalar(0));
                double mtx[9];
                double dist[8];
                mtx[0]  = RGBDinfo.RGB.fx; mtx[2]  = RGBDinfo.RGB.cx; mtx[4]  = RGBDinfo.RGB.fy; mtx[5]  = RGBDinfo.RGB.cy; mtx[8] = 1;
                dist[0] = RGBDinfo.RGB.k1; dist[1] = RGBDinfo.RGB.k2; dist[2] = RGBDinfo.RGB.p1; dist[3] = RGBDinfo.RGB.p2;
                dist[4] = RGBDinfo.RGB.k3; dist[5] = RGBDinfo.RGB.k4; dist[6] = RGBDinfo.RGB.k5; dist[7] = RGBDinfo.RGB.k6;
                memcpy(K.data, mtx, sizeof(double) * 9);
                memcpy(D.data, dist,sizeof(double) * 8);

                cv::Mat K_cam_1280_960 = (cv::Mat_<double>(3,3) << RGBDinfo.RGB.fx, 0, RGBDinfo.RGB.cx, 0, RGBDinfo.RGB.fy, RGBDinfo.RGB.cy, 0, 0, 1);
                cv::undistortPoints(src, dst, K, D, cv::noArray(), K_cam_1280_960);

                // 确认一下从模型出来的检测框会不会超过图片范围，有没有限制措施
                obj.col1 = std::max(0.0f, std::min(1280.0f, dst[0].x));
                obj.row1 = std::max(0.0f, std::min(960.0f,  dst[0].y));
                obj.col2 = std::max(0.0f, std::min(1280.0f, dst[1].x));
                obj.row2 = std::max(0.0f, std::min(960.0f,  dst[1].y));

            }
            else
            {
                obj.col1  = groundobjects->rect.x;
                obj.row1  = groundobjects->rect.y;
                obj.col2  = groundobjects->rect.x + groundobjects->rect.width;
                obj.row2  = groundobjects->rect.y + groundobjects->rect.height;
            }

            obj.score = groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].fconfidence;
            obj.cl    = groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].inlabel;

            if(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label >= EM_OUT_PMS 
            && groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label <= EM_OUT_DRIED_STAIN)
            {
                loc       = ecoDistanceCalProcess(obj, robot_lds_data, imu, 100);
                // if(abs(loc.x1 - loc.x2) > 20 || abs(loc.y1 - loc.y2) > 20)// 长或者宽大于20cm，结果输出为水渍
                // {
                //     groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label = EM_OUT_WATERSTAIN;
                // }
                if(abs(loc.x1 - loc.x2) * abs(loc.y1 - loc.y2) < 25)// 长或者宽大于20cm，结果输出为水渍
                {
                    groundobjects->bisobjects = false;
                    continue;
                }
                if(std::min(loc.x1, loc.x2) > 40)
                {
                    groundobjects->bisobjects = false;
                    continue;
                }
                if(abs(loc.x1 - loc.x2) > 40)
                {
                    loc.x1 = loc.x2 + 40;
                }
            }
            else
            {
                loc       = ecoDistanceCalProcess(obj, robot_lds_data, imu, model_infer_id);
            }
            
            // loc       = ecoDistanceCalProcess(obj, robot_lds_data, imu, model_infer_id);

            ////这里做个修改，将测距后的检测结果放到groundobjects->position中             
            cv::Point3f left_point_distance(loc.x1, loc.y1, -1);
            cv::Point3f right_point_distance(loc.x2, loc.y2, -1);
            groundobjects->position.push_back(left_point_distance);
            groundobjects->position.push_back(right_point_distance);

            if(1 == model_infer_id)
            {
                for(int nlds = 0; nlds < loc.bbox_lds_data.size(); nlds++)
                {
                    cv::Point3f lds_point_distance(-1, loc.bbox_lds_data[nlds], -1);
                    groundobjects->position.push_back(lds_point_distance);
                }
            }

            if(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_DOG)
            {
                if(std::min(loc.x1, loc.x2) > 200)
                {
                    groundobjects->bisobjects = false;
                    continue;
                }
            }

            if(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_WEIGHT_SCALE ||
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_BASE ||
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_CLOTH)
            {
                if((groundobjects->rect.y + groundobjects->rect.height) > 955)
                {
                    groundobjects->bisobjects = false;
                    continue;
                }
            }

#ifdef D_DEBUG

            std::cout << " x0 = " << groundobjects->position[0].x 
                        << " y0 = " << groundobjects->position[0].y << std::endl;

            std::cout << " x1 = " << groundobjects->position[1].x 
                        << " y1 = " << groundobjects->position[1].y << std::endl;

#endif

        }
    }



    // 红外目标检测标签的标签映射 与测距离（x，y，z）
    void topircamdectect(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu)
    {

        if(NULL == ecoinstanceobjectseg)
        {
            std::cout << "NULL == ecoinstanceobjectseg in topcamstataicdectect" << std::endl;     
        }

        if(ecoinstanceobjectseg->ecogroundobjects == NULL)
        {
            std::cout << "ecoinstanceobjectseg->ecogroundobjects == NULL in topcamstataicdectect" << std::endl;
        }


        ///////outecogroundobjects代表单帧全部目标检测结果---可能有多个目标检测结果
        EcoGroundObjectDects* outecogroundobjects = ecoinstanceobjectseg->ecogroundobjects;


        /////outecogroundobjects->ngroundobjectnum代表的是目标检测结果的个数
        for (size_t i = 0; i < outecogroundobjects->ngroundobjectnum; i++)
        {

            EcoGroundObjectDect*  groundobjects = &outecogroundobjects->ecogroundobject[i];

#ifdef D_DEBUG
            std:: cout << "prop = " << groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].fconfidence
            << "  label = " << groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].inlabel
            << "  xx = " << groundobjects->rect.x << "   yy = " << groundobjects->rect.y << std::endl;
#endif

            for (size_t j = 0; j < groundobjects->groundobjectsCls.nobjectsclsnum; j++)
            {
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[j].label = inlalbel2outlabel(model_infer_id, groundobjects->groundobjectsCls.ptrecogroundobjectscls[j].inlabel);
            }
            obj_detect obj;
            location loc;

            std::vector<cv::Point2f> src , dst;
            src.resize(4);
            dst.resize(4);
            ////这里src中的值可以做一个图片大小限制
            src[0] = cv::Point2f(groundobjects->rect.x, groundobjects->rect.y);                // 左上 (x, y)
            src[1] = cv::Point2f(groundobjects->rect.x + groundobjects->rect.width, groundobjects->rect.y); // 右上 (x+w, y)
            src[2] = cv::Point2f(groundobjects->rect.x + groundobjects->rect.width, groundobjects->rect.y + groundobjects->rect.height); // 右下 (x+w, y+h)
            src[3] = cv::Point2f(groundobjects->rect.x, groundobjects->rect.y + groundobjects->rect.height); // 左下 (x, y+h)

            cv::Mat K(3, 3, CV_64FC1, cv::Scalar(0));
            cv::Mat D(8, 1, CV_64FC1, cv::Scalar(0));
            double mtx[9];
            double dist[8];
            mtx[0]  = RGBDinfo.IR.fx; mtx[2]  = RGBDinfo.IR.cx; mtx[4]  = RGBDinfo.IR.fy; mtx[5]  = RGBDinfo.IR.cy; mtx[8] = 1;
            dist[0] = RGBDinfo.IR.k1; dist[1] = RGBDinfo.IR.k2; dist[2] = RGBDinfo.IR.p1; dist[3] = RGBDinfo.IR.p2;
            dist[4] = RGBDinfo.IR.k3; dist[5] = RGBDinfo.IR.k4; dist[6] = RGBDinfo.IR.k5; dist[7] = RGBDinfo.IR.k6;
            memcpy(K.data, mtx, sizeof(double) * 9);
            memcpy(D.data, dist,sizeof(double) * 8);

            cv::Mat K_cam_1280_720 = (cv::Mat_<double>(3,3) << RGBDinfo.IR.fx, 0, RGBDinfo.IR.cx, 0, RGBDinfo.IR.fy, RGBDinfo.IR.cy, 0, 0, 1);
            cv::undistortPoints(src, dst, K, D, cv::noArray(), K_cam_1280_720);

            // // 确认一下从模型出来的检测框会不会超过图片范围，有没有限制措施
            // cv::Rect rect;
            // rect.x = std::max(0.0f, std::min(1280.0f, dst[0].x));
            // rect.y = std::max(0.0f, std::min(720.0f,  dst[0].y));
            // rect.width = std::max(0.0f, std::min(1280.0f, dst[1].x)) - std::max(0.0f, std::min(1280.0f, dst[0].x));
            // rect.height = std::max(0.0f, std::min(720.0f,  dst[1].y)) - std::max(0.0f, std::min(720.0f,  dst[0].y));
            
            // //   增加红外图测距
            // float theta_x_2 = RGBDinfo.ir_rangle - atan((rect.y + rect.height - RGBDinfo.IR.cy) / RGBDinfo.IR.fy);
            // float x2       = RGBDinfo.ir_Height * tan(theta_x_2);

            // float theta_x_1 = RGBDinfo.ir_rangle - atan((rect.y - RGBDinfo.IR.cy) / RGBDinfo.IR.fy);
            // float x1       = RGBDinfo.ir_Height * tan(theta_x_1);

            // float y1 = (RGBDinfo.IR.cx - rect.x) * x1 / RGBDinfo.IR.fx;
            // float y2 = (RGBDinfo.IR.cx - (rect.x + rect.width)) * x2 / RGBDinfo.IR.fx;

            // ////这里做个修改，将测距后的检测结果放到groundobjects->position中             
            // cv::Point3f left_point_distance(x1, y1, -1);
            // cv::Point3f right_point_distance(x2, y2, -1);
            // groundobjects->position.push_back(left_point_distance);
            // groundobjects->position.push_back(right_point_distance);

            double min_x = std::min({dst[0].x, dst[1].x, dst[2].x, dst[3].x});
            double max_x = std::max({dst[0].x, dst[1].x, dst[2].x, dst[3].x});
            double min_y = std::min({dst[0].y, dst[1].y, dst[2].y, dst[3].y});
            double max_y = std::max({dst[0].y, dst[1].y, dst[2].y, dst[3].y});

            obj.col1 = std::max(0.0f, std::min(1279.0f, static_cast<float>(min_x)));
            obj.row1 = std::max(0.0f, std::min(719.0f,  static_cast<float>(min_y)));
            obj.col2 = std::max(0.0f, std::min(1279.0f, static_cast<float>(max_x)));
            obj.row2 = std::max(0.0f, std::min(719.0f,  static_cast<float>(max_y)));

            loc       = ecoDistanceCalProcess_ir(obj, robot_lds_data, imu, 100);

            cv::Point3f left_point_distance(loc.x1, loc.y1, -1);
            cv::Point3f right_point_distance(loc.x2, loc.y2, -1);
            groundobjects->position.push_back(left_point_distance);
            groundobjects->position.push_back(right_point_distance);

            if(loc.x1 == -1 || loc.y1 == -1 || int(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label) == 999)
            {
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label = EM_OUT_MID_AIR;
            }
            if(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_PM || groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_STAIN)
            {
                if(abs(loc.x1 - loc.x2) * abs(loc.y1 - loc.y2) < 25)
                {
                    groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label = EM_OUT_MID_AIR;
                    continue;
                }
            }
            if(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label == EM_OUT_PM)
            {
                if(std::min(loc.x1, loc.x2) > 45)
                {
                    groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label = EM_OUT_MID_AIR;
                    continue;
                }
            }
            else
            {
                if(std::min(loc.x1, loc.x2) > 40)
                {
                    groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label = EM_OUT_MID_AIR;
                    continue;
                }
            }

            if( groundobjects->rect.x < 10 || (groundobjects->rect.x + groundobjects->rect.width) > 1270)
            {
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label = EM_OUT_MID_AIR;
                continue;
            }

#ifdef D_DEBUG

            std::cout << " x0 = " << groundobjects->position[0].x 
                        << " y0 = " << groundobjects->position[0].y << std::endl;

            std::cout << " x1 = " << groundobjects->position[1].x 
                        << " y1 = " << groundobjects->position[1].y << std::endl;

#endif

        }
    }


    //计算每列的平均值
    void columnColsAverages(std::vector<std::vector<float>> &matrix_data, std::vector<float> &columnAverages)
    {
        if(matrix_data.size() == 0)
        {
            return;
        }
        size_t numRows = matrix_data.size();
        size_t numCols = matrix_data[0].size();

        std::vector<double> columnSums(numCols, 0.0);

        // 计算每列的总和
        for (size_t i = 0; i < numCols; ++i) {
            for (size_t j = 0; j < numRows; ++j) {
                columnSums[i] += matrix_data[j][i];
            }
        }

        // 计算每列的平均值
        // std::vector<float> columnAverages(numCols, 0.0);
        for (size_t i = 0; i < numCols; ++i) {
            columnAverages[i] = columnSums[i] / numRows;
        }

        return;
    }


    //计算每列的平均值
    void columnRowAverages(std::vector<float> &matrix_data, float &columnAverages)
    {
        if(matrix_data.size() == 0)
        {
            return;
        }
        size_t numCols = matrix_data.size();
        float columnSums = 0.0;

        // 计算每列的总和
        for (size_t j = 0; j < numCols; ++j) {
            columnSums += matrix_data[j];
        }

        columnAverages = columnSums / numCols;

        return;
    }


    // 计算线性回归的斜率
    double calculate_slope(const std::vector<double>& window) {
        int n = window.size();
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = i;
            double y = window[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        double denominator = n * sum_x2 - sum_x * sum_x;
        if (denominator == 0) return 0.0;
        return (n * sum_xy - sum_x * sum_y) / denominator;
    }

    // 合并区间函数
    std::vector<Interval> merge_intervals(const std::vector<Interval>& intervals) 
    {
        if (intervals.empty()) return {};
        std::vector<Interval> sorted_intervals = intervals;
        std::sort(sorted_intervals.begin(), sorted_intervals.end(),
            [](const Interval& a, const Interval& b) { return a.start < b.start; });
        
        std::vector<Interval> merged;
        merged.push_back(sorted_intervals[0]);
        
        for (size_t i = 1; i < sorted_intervals.size(); ++i) {
            Interval& last = merged.back();
            const Interval& current = sorted_intervals[i];
            
            if (current.start <= last.end + 1) {
                // 合并区间，更新结束位置为较大值
                last.end = std::max(last.end, current.end);
            } else {
                merged.push_back(current);
            }
        }
        return merged;
    }

    void findTrendRegression(std::vector<float> &data, int window_size, float slope_threshold, std::vector<TrendInterval> &interval_data) 
    {
        if(data.size() < window_size)
        {
            return;
        }
        std::vector<Interval> intervals_up;
        std::vector<Interval> intervals_down;

        enum class Trend { None, Up, Down, Flat };
        Trend current_trend = Trend::None;
        int start_idx = 0;
        double current_slope = 0.0;

        for (int i = 0; i <= data.size() - window_size; ++i) {
            std::vector<double> window(data.begin() + i, data.begin() + i + window_size);
            double slope = calculate_slope(window);

            Trend trend;
            if (slope > slope_threshold) {
                trend = Trend::Up;
            } else if (slope < -slope_threshold) {
                trend = Trend::Down;
            } else {
                trend = Trend::Flat;
            }

            if (trend != current_trend) {
                // 保存前一个趋势区间
                if (current_trend != Trend::None && current_trend != Trend::Flat) {
                    int end = i + window_size - 1;
                    if (current_trend == Trend::Up) {
                        intervals_up.emplace_back(start_idx, end, current_slope);
                    } else {
                        intervals_down.emplace_back(start_idx, end, current_slope);
                    }
                }
                // 更新当前趋势
                start_idx = i;
                if (trend == Trend::Flat) {
                    current_trend = Trend::None;
                } else {
                    current_trend = trend;
                }
                current_slope = slope;
            }
        }

        // 处理最后一个趋势区间
        if (current_trend != Trend::None && current_trend != Trend::Flat) {
            int end = data.size() - 1;
            if (current_trend == Trend::Up) {
                intervals_up.emplace_back(start_idx, end, current_slope);
            } else {
                intervals_down.emplace_back(start_idx, end, current_slope);
            }
        }
        // std::cout <<"intervals_:" << std::endl;
        // for(int ll = 0; ll < intervals_up.size(); ll++)
        // {
        //     std::cout << intervals_up[ll].start << " " << intervals_up[ll].end << " " << intervals_up[ll].slope << std::endl;
        // }
        // for(int ll = 0; ll < intervals_down.size(); ll++)
        // {
        //     std::cout << intervals_down[ll].start << " " << intervals_down[ll].end << " " << intervals_down[ll].slope << std::endl;
        // }

        // 合并区间
        auto merged_up = merge_intervals(intervals_up);
        auto merged_down = merge_intervals(intervals_down);

        // std::cout <<"merged_intervals_:" << std::endl;
        // for(int ll = 0; ll < merged_up.size(); ll++)
        // {
        //     std::cout << merged_up[ll].start << " " << merged_up[ll].end << " " << merged_up[ll].slope << std::endl;
        // }
        // for(int ll = 0; ll < merged_down.size(); ll++)
        // {
        //     std::cout << merged_down[ll].start << " " << merged_down[ll].end << " " << merged_down[ll].slope << std::endl;
        // }

        // 生成结果
        for (const auto& interval : merged_up) {
            int start = interval.start;
            int end = interval.end;
            if (start >= data.size() || end >= data.size() || start > end) continue;

            auto max_it = max_element(data.begin() + start, data.begin() + end + 1);
            auto min_it = min_element(data.begin() + start, data.begin() + end + 1);
            double value = *max_it - *min_it;

            interval_data.push_back({value, start, end, "up"});
        }

        for (const auto& interval : merged_down) {
            int start = interval.start;
            int end = interval.end;
            if (start >= data.size() || end >= data.size() || start > end) continue;

            auto max_it = max_element(data.begin() + start, data.begin() + end + 1);
            auto min_it = min_element(data.begin() + start, data.begin() + end + 1);
            double value = *max_it - *min_it;

            interval_data.push_back({value, start, end, "down"});
        }

        // 按起始位置排序
        std::sort(interval_data.begin(), interval_data.end(),
            [](const TrendInterval& a, const TrendInterval& b) { return a.start < b.start; });

        return ;

    }

    //分割检测结果与结构光数据进行融合矫正
    void carpetFreespacePointToLinesaserFusion(cv::Mat &carpet_freespace_mask, EcoAInterfaceSlData_t& SLSData, CamerainnerInfo& paramCam, bool &carpet_mask_valid_flag,std::vector<std::vector<float>> &linelaser_to_cam,std::vector<float> &linelaser_ground_points_colmean,std::vector<std::vector<float>> &linelaser_z_grayscale_average)
    {
        if(SLSData.idx == 1)
        {
            carpet_mask_valid_flag = false;
            return;
        }
        std::vector<float>  Tcl = {0, -1, 0, 0, 0, -1, 1, 0, 0, 11.0, 66.5, -145};

        // carpet_mask腐蚀膨胀一下,256*192大小
        cv::Mat carpet_mask = cv::Mat::zeros(carpet_freespace_mask.rows, carpet_freespace_mask.cols, CV_8UC1);
        bool carpet_exist_flag = false;
        for(int row = 0; row < carpet_freespace_mask.rows; row++)
        {
            for(int col = 0; col < carpet_freespace_mask.cols; col++)
            {
                if (carpet_freespace_mask.at<uchar>(row, col) >= 200)
                {
                    carpet_mask.at<uchar>(row, col) = 255;
                    carpet_exist_flag = true;
                }
            }
        }
        // cv::namedWindow("carpet", 0);
        // cv::imshow("carpet", carpet_mask);
        // cv::waitKey(0);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(15, 15));
        cv::dilate(carpet_mask, carpet_mask, element);
        cv::erode(carpet_mask, carpet_mask, element);

        int carpet_mask_cols = carpet_mask.cols;
        int carpet_mask_rows = carpet_mask.rows;
    //        std::cout << "carpet_mask_cols:" << carpet_mask_cols << " " << carpet_mask_rows << std::endl;
    //        std::cout << "paramCam:" << paramCam.fx << " " << paramCam.fy << " " << paramCam.cx << " " << paramCam.cy << std::endl;
        int z_scale = 100;
        float point_x = 0;
        float point_y = 0;
        float point_z = 0;
        float point_grayscale = 0;
        std::vector<float> point_cam(3,0);
        std::vector<std::vector<float>> linelaser_carpet_points;
        std::vector<std::vector<float>> linelaser_nonecarpet_points;
        std::vector<float> linelaser_z_points,linelaser_grayscale_points;
        int col = 0;
        int row = 0;

        int valid_idx = 0;
        for(int ll = 0; ll < SL_POINT_CNT; ll++)
        {
            point_x = SLSData.slPoint[ll].x;
            point_y = SLSData.slPoint[ll].y;
            point_z = SLSData.slPoint[ll].z;
            point_grayscale = SLSData.slPoint[ll].grayscales;
            // std::cout << "SLSData: " << point_x  << " " << point_y << " " << point_z << " " << point_grayscale << std::endl;
            if(point_x == 0 && point_y == 0 && point_z == 0 && point_grayscale == 0)
            {
                continue;
            }
    //            std::cout << "SLSData: " << point_x  << " " << point_y << " " << point_z << " " << point_grayscale << std::endl;
            point_cam = {Tcl[0] * point_x + Tcl[1] * point_y + Tcl[2] * point_z + Tcl[9],
                            Tcl[3] * point_x + Tcl[4] * point_y + Tcl[5] * point_z + Tcl[10],
                            Tcl[6] * point_x + Tcl[7] * point_y + Tcl[8] * point_z + Tcl[11]};
            if(point_cam[2] != 0)
            {
                // 1280*960尺寸上映射到256*192上
                col = int(((paramCam.fx) * point_cam[0] / point_cam[2] + (paramCam.cx) + 0.5)/5);
                row = int(((paramCam.fy) * point_cam[1] / point_cam[2] + (paramCam.cy) + 0.5)/5);
                if(col >= 0 && col < carpet_mask_cols && row >= 0 && row < carpet_mask_rows)
                {
                    std::vector<float> vec1 = {row, col, point_z * z_scale, point_grayscale, valid_idx};
                    std::vector<float> vec2 = {row, col,point_x, point_y, point_z,point_grayscale, valid_idx};
                    linelaser_to_cam.push_back(vec2);
                    valid_idx += 1;
                    if(carpet_mask.at<uchar>(row, col) >= 10)
                    {
                        linelaser_carpet_points.push_back(vec1);
                    }else{
                        linelaser_nonecarpet_points.push_back(vec1);
                    }

                    linelaser_z_points.push_back(point_z * z_scale);
                    linelaser_grayscale_points.push_back(point_grayscale);
                }
            }
        }
        // std::cout << "linelaser_carpet_points:" << linelaser_z_points.size() << " " << linelaser_carpet_points.size() << " " << linelaser_nonecarpet_points.size() << " "<<std::endl;
        // std::cout << "pre_mean_z_: " << linelaser_ground_points_colmean[0] << " " << linelaser_ground_points_colmean[1] << std::endl;

        std::vector<TrendInterval> peak_valley_intervals,peak_valley_intervals_grayscale;
        findTrendRegression(linelaser_z_points, 9, 40, peak_valley_intervals);
        findTrendRegression(linelaser_grayscale_points, 5, 6.5, peak_valley_intervals_grayscale);
        // std::cout << "peak_valley_intervals_size:" << peak_valley_intervals.size() << " " << peak_valley_intervals_grayscale.size() << std::endl;

        if(linelaser_carpet_points.size() >= 5)
        {
            int valid_linelaser_point_size = linelaser_carpet_points.size();
            int valid_linelaser_start_col = linelaser_carpet_points[0][4];
            int valid_linelaser_end_col = linelaser_carpet_points[valid_linelaser_point_size - 1][4];
            std::vector<float> linelaser_carpet_points_colmean(5, 0.0);
            std::vector<float> linelaser_nonecarpet_points_colmean(5, 0.0);
            columnColsAverages(linelaser_carpet_points,linelaser_carpet_points_colmean);
            columnColsAverages(linelaser_nonecarpet_points,linelaser_nonecarpet_points_colmean);

            // std::vector<TrendInterval> peak_valley_intervals;
            // findTrendRegression(linelaser_z_points, 9, 40, peak_valley_intervals);
            bool peak_valley_interval_flag = false;
            //判断结构光地毯语义中心点：右侧最近的下降区域，左侧最近的上升区域是否存在，是否符合要求
            float valid_linlaser_col_center = linelaser_carpet_points_colmean[4];

            // std::cout << "valid_linelaser_point_size: " << linelaser_z_points.size() << " " << valid_linelaser_point_size << " " << valid_linelaser_start_col << " " << valid_linelaser_end_col << std::endl;

            // std::cout << "peak_valley_intervals_size:" << peak_valley_intervals.size() << std::endl;
            // std::cout << "valid_linlaser_col_center: " << valid_linlaser_col_center << " carpet: " <<
            // linelaser_carpet_points_colmean[0] << " " <<linelaser_carpet_points_colmean[1] << " " <<linelaser_carpet_points_colmean[2] << " " <<
            // linelaser_carpet_points_colmean[3] << " " << linelaser_carpet_points_colmean[4] << " nonecarpet: "
            // <<linelaser_nonecarpet_points_colmean[0] << " " <<
            // linelaser_nonecarpet_points_colmean[1] << " " <<
            // linelaser_nonecarpet_points_colmean[2] << " " <<
            // linelaser_nonecarpet_points_colmean[3] << " " <<
            // linelaser_nonecarpet_points_colmean[4] <<std::endl;
            if(peak_valley_intervals.size() < 6)
            {
                for(int ll = 0; ll < peak_valley_intervals.size(); ll++)
                {
                    TrendInterval peak_valley_interval = peak_valley_intervals[ll];
                    std::cout << peak_valley_interval.start << " " << peak_valley_interval.end << " " << peak_valley_interval.value << " " << peak_valley_interval.trend <<std::endl;

                    if((valid_linlaser_col_center < peak_valley_interval.start || abs(valid_linlaser_col_center - peak_valley_interval.start) < 5)
                        && peak_valley_interval.trend == "down"
                        && (abs(valid_linlaser_col_center + valid_linelaser_point_size/2 -peak_valley_interval.start) < 18 || abs(valid_linelaser_end_col - peak_valley_interval.start) < 18)
                        && peak_valley_interval.value >= 265
                        && peak_valley_interval.value <= 4000
                        && ((linelaser_carpet_points_colmean[3] - linelaser_nonecarpet_points_colmean[3]) >= 2.3 && (linelaser_carpet_points_colmean[3] - linelaser_ground_points_colmean[1]) >= 3)
                        && linelaser_nonecarpet_points_colmean[2] <= 1000
                        && linelaser_carpet_points_colmean[3] <= 100
                        && (linelaser_carpet_points_colmean[2] -  linelaser_nonecarpet_points_colmean[2]) >= 150
                        && abs(linelaser_ground_points_colmean[0]) <= 900)
                        {
                            peak_valley_interval_flag = true;
                            // std::cout << "down" << std::endl;
                            break;
                        }

                    if((valid_linlaser_col_center > peak_valley_interval.end || abs(valid_linlaser_col_center - peak_valley_interval.end) < 5)
                        && peak_valley_interval.trend == "up"
                        && (abs(valid_linlaser_col_center - valid_linelaser_point_size/2 -peak_valley_interval.end) < 18 || abs(valid_linelaser_start_col - peak_valley_interval.end) < 18)
                        && peak_valley_interval.value >= 265
                        && peak_valley_interval.value <= 4000
                        && ((linelaser_carpet_points_colmean[3] - linelaser_nonecarpet_points_colmean[3]) >= 2.3 && (linelaser_carpet_points_colmean[3] - linelaser_ground_points_colmean[1]) >= 3)
                        && linelaser_nonecarpet_points_colmean[2] <= 1000
                        && linelaser_carpet_points_colmean[3] <= 100
                        && (linelaser_carpet_points_colmean[2] -  linelaser_nonecarpet_points_colmean[2]) >= 150
                        && abs(linelaser_ground_points_colmean[0]) <= 900)
                        {
                            peak_valley_interval_flag = true;
                            // std::cout << "up" << std::endl;
                            break;
                        }
                }
            }

            float valid_carpet_point_rate = valid_linelaser_point_size / ((valid_linelaser_end_col - valid_linelaser_start_col + 1) * 1.0);
            float valid_linelaser_point_rate = (valid_linelaser_end_col - valid_linelaser_start_col + 1) / ((linelaser_carpet_points.size() + linelaser_nonecarpet_points.size()) * 1.0);
            // std::cout << "valid_carpet_point_rate: " << valid_carpet_point_rate << " " << valid_linelaser_point_rate << std::endl;
            if(linelaser_carpet_points_colmean[2] >= 300 && linelaser_carpet_points_colmean[2] <= 4000
                && valid_carpet_point_rate >= 0.8 && valid_linelaser_point_rate >= 0.8
                && linelaser_carpet_points_colmean[3] <= 100 && linelaser_carpet_points_colmean[2] > linelaser_nonecarpet_points_colmean[2]
                && linelaser_carpet_points_colmean[3] - linelaser_ground_points_colmean[1] >= 3
                && abs(linelaser_ground_points_colmean[0]) <= 900)
                {
                    carpet_mask_valid_flag = true;
                    // std::cout << "tiaojian1" << std::endl;
                }else
                {
                    if (peak_valley_interval_flag)
                    {
                        carpet_mask_valid_flag = true;
                        // std::cout << "tiaojian2" << std::endl;
                    }else
                    {
                        carpet_mask_valid_flag = false;
                    }
                }
        }else
        {
            if(carpet_exist_flag)
            {
                carpet_mask_valid_flag = false;
            }else{
                carpet_mask_valid_flag = false;
            }
        }

        if(linelaser_carpet_points.size() == 0 && peak_valley_intervals.size() == 0 && peak_valley_intervals_grayscale.size() == 0)
        {
            float linelaser_z_average = 0.0, linelaser_grayscale_average = 0.0;
            columnRowAverages(linelaser_z_points, linelaser_z_average);
            columnRowAverages(linelaser_grayscale_points, linelaser_grayscale_average);
            linelaser_z_grayscale_average.push_back({linelaser_z_average,linelaser_grayscale_average});
            if(linelaser_z_grayscale_average.size() == 3)
            {
                columnColsAverages(linelaser_z_grayscale_average,linelaser_ground_points_colmean);
                std::vector<std::vector<float>> pre_linelaser_z_grayscale_average;
                pre_linelaser_z_grayscale_average.push_back(linelaser_z_grayscale_average[1]);
                pre_linelaser_z_grayscale_average.push_back(linelaser_z_grayscale_average[2]);
                linelaser_z_grayscale_average.clear();
                linelaser_z_grayscale_average = pre_linelaser_z_grayscale_average;
            }
        }
        // std::cout << "pre_mean_z: " << linelaser_ground_points_colmean[0] << " " << linelaser_ground_points_colmean[1] << std::endl;
        // std::cout << "carpet_mask_valid_flag:" << carpet_mask_valid_flag << std::endl;

        return;
    }

    void secondVerificationCheckCarpetMaskVdlidFalg(cv::Mat &carpet_linelaser_map_probability, std::vector<cv::Point> &carpet_linelaser_valid_pose_vec, float pose_x, float pose_y, float theta, EcoInstanceObjectSeg *ecoinstanceobjectseg, bool &carpet_mask_valid_flag)
    {
        int zero_point_x = 30000, zero_point_y = 30000, pixel_size = 50;
        if(carpet_mask_valid_flag)
        {
            float dis_x = 0.0, dis_y = 0.0;
            std::vector<cv::Point> target_points;
            target_points.clear();
            for (size_t i = 0; i < ecoinstanceobjectseg->maskdata.size(); i++)
            {
                dis_x = ecoinstanceobjectseg->maskdata[i].keypoint.x;
                dis_y = ecoinstanceobjectseg->maskdata[i].keypoint.y;
                if(dis_x * 10 > 450)
                {
                    continue;
                }
                float pt_x = 0,pt_y = 0;
                int point_x = 0,point_y = 0;
                pt_x = (pose_x + float(dis_x * 10 + 170) * std::cos(theta) - float(dis_y * 10) * std::sin(theta));
                pt_y = (pose_y + float(dis_x * 10 + 170) * std::sin(theta) + float(dis_y * 10) * std::cos(theta));
                point_x = int((zero_point_x + pt_x) / pixel_size + 0.5);
                point_y = int((zero_point_y + pt_y) / pixel_size + 0.5);
                std::vector<cv::Point>::iterator itt = find(target_points.begin(), target_points.end(), cv::Point(point_x, point_y));
                if (itt != target_points.end())
                {
                    continue;
                }
                target_points.push_back(cv::Point(point_x, point_y));
                if(point_x >= 0 && point_x < 1200 && point_y >= 0 && point_y < 1200)
                {
                    carpet_linelaser_map_probability.at<uchar>(point_y, point_x) += 1;
                }
            }
            // carpet_linelaser_valid_pose_vec.push_back(cv::Point(pose_x, pose_y));
        }else
        {
            int robot_x = int((zero_point_x + pose_x) / pixel_size + 0.5);
            int robot_y = int((zero_point_y + pose_y) / pixel_size + 0.5);
            if(carpet_linelaser_map_probability.at<uchar>(robot_y, robot_x) >= 3)
            {
                carpet_mask_valid_flag = true;
                float dis_x = 0.0, dis_y = 0.0;
                std::vector<cv::Point> target_points;
                target_points.clear();
                for (size_t i = 0; i < ecoinstanceobjectseg->maskdata.size(); i++)
                {
                    dis_x = ecoinstanceobjectseg->maskdata[i].keypoint.x;
                    dis_y = ecoinstanceobjectseg->maskdata[i].keypoint.y;
                    if(dis_x * 10 > 450)
                    {
                        continue;
                    }
                    float pt_x = 0,pt_y = 0;
                    int point_x = 0,point_y = 0;
                    pt_x = (pose_x + float(dis_x * 10 + 170) * std::cos(theta) - float(dis_y * 10) * std::sin(theta));
                    pt_y = (pose_y + float(dis_x * 10 + 170) * std::sin(theta) + float(dis_y * 10) * std::cos(theta));
                    point_x = int((zero_point_x + pt_x) / pixel_size + 0.5);
                    point_y = int((zero_point_y + pt_y) / pixel_size + 0.5);
                    std::vector<cv::Point>::iterator itt = find(target_points.begin(), target_points.end(), cv::Point(point_x, point_y));
                    if (itt != target_points.end())
                    {
                        continue;
                    }
                    target_points.push_back(cv::Point(point_x, point_y));
                    if(point_x >= 0 && point_x < 1200 && point_y >= 0 && point_y < 1200)
                    {
                        carpet_linelaser_map_probability.at<uchar>(point_y, point_x) += 1;
                    }
                }
            }
            // else{
            //     double dis = 0;
            //     for(int ii = carpet_linelaser_valid_pose_vec.size() - 1; ii > -1; ii--)
            //     {
            //         dis = sqrt((pose_x - carpet_linelaser_valid_pose_vec[ii].x) * (pose_x - carpet_linelaser_valid_pose_vec[ii].x) + 
            //         (pose_y - carpet_linelaser_valid_pose_vec[ii].y) * (pose_y - carpet_linelaser_valid_pose_vec[ii].y));
            //         if(dis <= 450)
            //         {
            //             carpet_mask_valid_flag = true;
            //             // carpet_valid_pose_vec.push_back(cv::Point(SLSData.status.x, SLSData.status.y));
            //             break;
            //         }
            //     }
            // }
        }
    }



    // 语义分割和关键点检测的标签映射 与测距离（x，y，z）
    void topcamstataicseg(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu, cv::Mat& single_rug_mask)
    {

        if(NULL == ecoinstanceobjectseg)
        {
            std::cout << "NULL == ecoinstanceobjectseg in topcamstataicseg" << std::endl;     
        }

        int  nsefnum  = -1;     //  BEV 视角地毯面积统计
        bool bTASSELS = false;  //  流苏标志位

        location loc_before;
        loc_before.x1 = -1;
        loc_before.y1 = -1;
        loc_before.x2 = -1;
        loc_before.y2 = -1;
        // 地毯数据过滤
        cv::Point3f max_keypoint(cv::Point3f(0,0,0));
        cv::Point3f mix_keypoint(cv::Point3f(1000,1000,0));
        ///

        cv::Mat rug_mask_u = cv::Mat(45, 80, CV_8UC1, cv::Scalar(255));

        for (size_t i = 0; i < ecoinstanceobjectseg->maskdata.size(); i++)
        {
            ////////将模型出来的inlabel转成对外输出的label
            ecoinstanceobjectseg->maskdata[i].label = inlalbel2outlabel(model_infer_id, ecoinstanceobjectseg->maskdata[i].inlabel);     
/***********  按着距离将目标过滤    ******************************************************************************************/
 

            obj_detect obj;
            location loc;

            obj.col1  = ecoinstanceobjectseg->maskdata[i].mappos.x;
            obj.row1  = ecoinstanceobjectseg->maskdata[i].mappos.y;
            obj.col2  = ecoinstanceobjectseg->maskdata[i].mappos.x;
            obj.row2  = ecoinstanceobjectseg->maskdata[i].mappos.y;
            obj.score = ecoinstanceobjectseg->maskdata[i].fconfidence;
            obj.cl    = ecoinstanceobjectseg->maskdata[i].inlabel;     /////这个cl要的应该是每个模型直接输出的类别id中的inlabel，没有经过映射的

            loc = ecoDistanceCalProcess(obj,  robot_lds_data, imu, model_infer_id);

            // ecoinstanceobjectseg->maskdata[i].keypoint.x = loc.x1;
            // ecoinstanceobjectseg->maskdata[i].keypoint.y = loc.y1;
            ecoinstanceobjectseg->maskdata[i].keypoint.x = loc.x2;
            ecoinstanceobjectseg->maskdata[i].keypoint.y = loc.y2;

            if(loc.x2 > 90 || loc.x2 < 3 || int(ecoinstanceobjectseg->maskdata[i].label) == 999)
            {
                ecoinstanceobjectseg->maskdata[i].bistrue = false;
                continue;
            }
            if (model_infer_id == 3)
            {
                // 地毯以及流苏 按着 4 * 4 的范围过滤
                if(ecoinstanceobjectseg->maskdata[i].bistrue && (ecoinstanceobjectseg->maskdata[i].label == EM_OUT_CARPET || ecoinstanceobjectseg->maskdata[i].label == EM_OUT_CARPET_EDGE))
                {
                    if(ecoinstanceobjectseg->maskdata[i].keypoint.x > 25){
                        int row = (int)ecoinstanceobjectseg->maskdata[i].keypoint.x / 4 + 7;
                        int col = (int)ceil((0 - ecoinstanceobjectseg->maskdata[i].keypoint.y) / 4) + 40;
                        if (row >= 0 && row < single_rug_mask.rows && col >= 0 && col < single_rug_mask.cols) 
                        {
                            if(single_rug_mask.at<uchar>(row, col) == 125)
                            {
                                ecoinstanceobjectseg->maskdata[i].bistrue = false;
                            }
                            else
                            {
                                single_rug_mask.at<uchar>(row, col) = 125;
                                nsefnum += 16;
                            }
                        }
                    }   
                    else
                    {
                        int row = (int)ecoinstanceobjectseg->maskdata[i].keypoint.x / 2;
                        int col = (int)ceil((0 - ecoinstanceobjectseg->maskdata[i].keypoint.y) / 2) + 40;
                        if (row >= 0 && row < single_rug_mask.rows && col >= 0 && col < single_rug_mask.cols) 
                        {
                            if(single_rug_mask.at<uchar>(row, col) == 125)
                            {
                                ecoinstanceobjectseg->maskdata[i].bistrue = false;
                            }
                            else
                            {
                                single_rug_mask.at<uchar>(row, col) = 125;
                                nsefnum += 4;
                            }
                        }
                    }                   
                }
                // u形椅 bev过滤
                if(ecoinstanceobjectseg->maskdata[i].bistrue && (ecoinstanceobjectseg->maskdata[i].label == EM_OUT_UCHAIR_BASE))
                {
                    int row = (int)ecoinstanceobjectseg->maskdata[i].keypoint.x / 2;
                    int col = (int)ceil((0 - ecoinstanceobjectseg->maskdata[i].keypoint.y) / 2) + 40;
                    // 添加边界检查
                    if (row >= 0 && row < rug_mask_u.rows && col >= 0 && col < rug_mask_u.cols) 
                    {
                        if(rug_mask_u.at<uchar>(row, col) == 10)
                        {
                            ecoinstanceobjectseg->maskdata[i].bistrue = false;
                        }
                        else
                        {
                            rug_mask_u.at<uchar>(row, col) = 10;
                        }
                    }                     
                }
            }

            // // 同一个点位 地毯（125） 优先级大于 无语义接地线
            // if(ecoinstanceobjectseg->maskdata[i].label == EM_OUT_BASE 
            // && single_rug_mask.at<uchar>((int)ecoinstanceobjectseg->maskdata[i].keypoint.x/4, (int)(0-ecoinstanceobjectseg->maskdata[i].keypoint.y)/4 + 40) == 125)
            // {
            //     ecoinstanceobjectseg->maskdata[i].bistrue = false;
            // }

            // // 有流苏的小地毯也保留
            // if(ecoinstanceobjectseg->maskdata[i].label == EM_OUT_TASSELS)
            // {
            //     bTASSELS = true;
            // }

    #ifdef D_DEBUG
            //　获取当前时间并计算运行时间
            std::cout << "  x = " << ecoinstanceobjectseg->maskdata[i].keypoint.x 
                      << "  y = " << ecoinstanceobjectseg->maskdata[i].keypoint.y << std::endl;
            
            std::cout << "length = " << ecoinstanceobjectseg->maskdata[i].mappos.x << std::endl;
            std::cout << "label = " << ecoinstanceobjectseg->maskdata[i].label  << "   prop = " << ecoinstanceobjectseg->maskdata[i].fconfidence << std::endl;
    #endif
        
        }

        // // 单帧地毯面积小于320cm*cm 且无流苏则 删除单帧地毯结果
        // if(nsefnum < 320 && !bTASSELS && model_infer_id == 3)
        // {
        //     ecoinstanceobjectseg->maskdata.clear();
        // }

    }


    void topcamstataiccls(const int model_infer_id, EcoInstanceObjectSeg *ecoinstanceobjectseg, CameraInfo& RGBDinfo, std::vector<std::vector<float>>& robot_lds_data, std::vector<int> imu)
    {

        if(NULL == ecoinstanceobjectseg)
        {
            std::cout << "NULL == ecoinstanceobjectseg in topcamstataicdectect" << std::endl;     
        }

        if(ecoinstanceobjectseg->ecogroundobjects == NULL)
        {
            std::cout << "ecoinstanceobjectseg->ecogroundobjects == NULL in topcamstataicdectect" << std::endl;
        }

        ///////outecogroundobjects代表单帧全部目标检测结果---可能有多个目标检测结果
        EcoGroundObjectDects* outecogroundobjects = ecoinstanceobjectseg->ecogroundobjects;

        /////outecogroundobjects->ngroundobjectnum代表的是目标检测结果的个数
        for (size_t i = 0; i < outecogroundobjects->ngroundobjectnum; i++)
        {

            EcoGroundObjectDect*  groundobjects = &outecogroundobjects->ecogroundobject[i];
            for (size_t j = 0; j < groundobjects->groundobjectsCls.nobjectsclsnum; j++)
            {
                groundobjects->groundobjectsCls.ptrecogroundobjectscls[j].label = inlalbel2outlabel(model_infer_id, groundobjects->groundobjectsCls.ptrecogroundobjectscls[j].inlabel);
            }
            
            obj_detect obj;
            location loc;
            loc.x1 = 10;
            loc.y1 = 10;
            loc.x2 = 10;
            loc.y2 = 10;
            cv::Point3f left_point_distance(loc.x1, loc.y1, -1);
            cv::Point3f right_point_distance(loc.x2, loc.y2, -1);
            groundobjects->position.push_back(left_point_distance);
            groundobjects->position.push_back(right_point_distance);
            if(loc.x1 == -1 || loc.y1 == -1 || int(groundobjects->groundobjectsCls.ptrecogroundobjectscls[0].label) == 999)
            {
                groundobjects->bisobjects = false;
            }

        }

    }

static void inlalbel2outobjprops(EcoObjectProps& objectprop)
{
    switch (objectprop.inlabel)
    {
    case 0:
        objectprop.direction = EM_OUT_FRONT;
        break;
    case 1:
        objectprop.direction = EM_OUT_AFTER;
        break;
    case 2:
        objectprop.shape = EM_OUT_RECT;
        break;
    case 3:
        objectprop.shape = EM_OUT_CIRCLE;
        break;    
    default:
        objectprop.direction = EM_OTHERDIRECTIONS;
        objectprop.shape     = EM_OTHERSHAPES;
        break;
    }

}



    // 目标类型转换
    static EcoAILabels inlalbel2outlabel(const int model_infer_id, int &inlabel)
    {  

        if (0 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_BASE;
                break;
            case 1:
                return EM_OUT_TRASH_CAN;
                break;
            case 2:
                return EM_OUT_CLOTH;
                break;
            case 3:
                return EM_OUT_SHOES;
                break;
            case 4:
                return EM_OUT_WEIGHT_SCALE;
                break;
            case 5:
                return EM_OUT_PET_POOP;
                break;
            case 6:
                return EM_OUT_LINE;
                break;

            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        // 家具目标检测
        if (1000 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_DOOR_FRAME;
                break;
            case 1:
                return EM_OUT_SOFA;
                break;
            case 2:
                return EM_OUT_TABLE;
                break;
            case 3:
                return EM_OUT_DINING_TABLES_CHAIRS;
                break;
            case 4:
                return EM_OUT_TEA_TABLE;
                break;
            case 5:
                return EM_OUT_TV_CAB;
                break;
            case 6:
                return EM_OUT_TV;
                break;
            case 7:
                return EM_OUT_BED;
                break;
            case 8:
                return EM_OUT_BEDSIDE_TABLE;
                break;
            case 9:
                return EM_OUT_CLOSETOOL;
                break;
            case 10:      // 借用饮食盆当椅子
                return EM_OUT_DIET;
                break;
            case 11:
                return EM_OUT_BED;
                break;
            case 12:
                return EM_OUT_SOFA;
                break;
            case 13:
                return EM_OUT_WASHER;
                break;
            case 14:
                return EM_OUT_FRIDGE;
                break;
            case 15:
                return EM_OUT_CABINET;
                break;
            case 16:
                return EM_OUT_CUPBOARD;
                break;
            case 18:
                return EM_OUT_WARDROBE;
                break;
            case 19:
                return EM_OUT_MEAL_SIDE_CAB;
                break;
            case 20:
                return EM_OUT_SHOE_CAB;
                break;
            case 21:
                return EM_OUT_SHOE_CAB;
                break;
            case 22:
                return EM_OUT_FLOOR_MIRROR;
                break;
            case 23:
                return EM_OUT_ISLAND_PLATFORM;
                break;
            case 25:
                return EM_OUT_GREEN_PLANTS;
                break;
            case 26:
                return EM_OUT_PIANO;
                break;
            case 27:
                return EM_OUT_AIR_CONDITION;
                break;
            case 28:
                return EM_OUT_BOOKCASE;
                break;
            case 29:
                return EM_OTHERLABELS;
                break;
            case 30:
                return EM_OUT_OVEN;
                break;
            case 31:
                return EM_OUT_GRATE;
                break;
            case 32:
                return EM_OUT_DISH_WASHER;
                break;
            case 33:
                return EM_OUT_DIET;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }  

        // 家具目标检测
        if (1 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_DOOR_FRAME;
                break;
            case 1:
                return EM_OUT_SOFA;
                break;
            case 2:
                return EM_OUT_DINING_TABLES_CHAIRS;
                break;
            case 3:
                return EM_OUT_TEA_TABLE;
                break;
            case 4:
                return EM_OUT_TV_CAB;
                break;
            case 5:
                return EM_OUT_BED;
                break;
            case 6:
                return EM_OUT_BEDSIDE_TABLE;
                break;
            case 7:
                return EM_OUT_CLOSETOOL;
                break;
            case 8:
                return EM_OUT_WASHER;
                break;
            case 9:
                return EM_OUT_FRIDGE;
                break;
            case 10:
                return EM_OUT_CUPBOARD;
                break;
            case 11:
                return EM_OUT_WARDROBE;
                break;
            case 12:
                return EM_OUT_SHOE_CAB;
                break;
            case 13:
                return EM_OUT_FLOOR_MIRROR;
                break;
            case 14:
                return EM_OUT_GREEN_PLANTS;
                break;
            case 15:
                return EM_OUT_AIR_CONDITION;
                break;
            case 16:
                return EM_OUT_BOOKCASE;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }  


        //人形
        if (2 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_PERSON;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }


        // 地毯和uchair分割---其中包含检测
        if (3 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_BACKGROUND;
                break;
            case 1:
                return EM_OUT_GROUND;
                break;
            case 2:  //  地毯 mask 面
                return EM_OUT_CARPET;
                break;
            case 3:  //  地毯 mask 边缘
                return EM_OUT_CARPET;
                break;
            case 4:
                return EM_OUT_UCHAIR_BASE;
                break;
            // case 5:
            //     return EM_OUT_CARPET;
            //     break;
            case 6:
                return EM_OUT_TASSELS;
                break;
            case 9:
                return EM_OUT_WEIGHT_SCALE;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }  

        // 通用障碍物
        if (10 == model_infer_id)
        {
            switch (inlabel)
            {
            case 7:
                return EM_OUT_BASE;
                break;
            case 8:
                return EM_OUT_LINE;
                break;
            case 9:
                return EM_OUT_CARPET_EDGE;
                break;
            case 11:
                return EM_OUT_CILL;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }  

        // 颗粒物
        if (6 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_PM;    // 单（堆）颗粒物
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        // 水渍
        if (7 == model_infer_id)
        {
            switch (inlabel)
            {
            case 1:
                return EM_OUT_STAIN; // 混合态污渍 EM_OUT_MIXEDSTATE
                break;
            case 2:
                return EM_OUT_STAIN; // 重污渍
                break;
            case 3:
                return EM_OUT_STAIN; // 清水 EM_OUT_WATERSTAIN
                break;
            case 4:
                return EM_OUT_WATERSTAIN; // 地插借用EM_OUT_WATERSTAIN清水标签用来数据闭环
                break;
            case 50:
                return EM_OUT_PMS;  // 轻污渍 EM_OUT_PMS 只有开普勒 SE
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        // 污渍
        if (9 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_STAIN;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }
    
        if (4 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_UNDERBED;
                break;
            case 1:
                return EM_OUT_NOUNDERBED;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        if (5 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_FLOOR;
                break;
            case 1:
                return EM_OUT_FLOOR_TILE;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        // 红外污渍识别
        if (21 == model_infer_id)
        {
            switch (inlabel)
            {
                case 0:
                    return EM_OUT_PM;
                    break;
                case 1:
                    return EM_OUT_MIXEDSTATE;
                    break;
                case 2:
                    return EM_OUT_STAIN;
                    break;
                case 50:
                    return EM_OUT_MID_AIR;
                default:
                    return EM_OTHERLABELS;
                    break;
            }
        }

        // 宠物识别
        if (11 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_DOG;
                break;
            case 50:
                return EM_OUT_CAT;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        } 

        // 障碍物识别
        if (12 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_WEIGHT_SCALE;
                break;
            case 1:
                return EM_OUT_BASE;
                break;
            case 2:
                return EM_OUT_CLOTH;
                break;
            case 3:
                return EM_OUT_PET_POOP;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        } 

        // 毛絮识别
        if (13 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_LINT;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        // 干涸污渍识别
        if (14 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_DRIED_STAIN;
                break;
            case 3:
                return EM_OUT_WATERSTAIN;
                break;
            case 4:
                return EM_OUT_DRIED_STAIN;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        }

        if (15 == model_infer_id)
        {
            switch (inlabel)
            {
            case 0:
                return EM_OUT_PURE_TILE;
                break;
            default:
                return EM_OTHERLABELS;
                break;
            }
        } 

        return EM_OTHERLABELS;
    }



    ////家具检测----yolov7
    void yolov7_post_process(EcoRknnModelParams& modelparams, rknn_output outputs[3],  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_){
        /////在这里把我传进来的数据转成rknn_app_context_t、letterbox_t、object_detect_result_list等形式后调用

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold  = iou_out_Threshold_of_each_[0][0];
        conf_threshold = 0.25;

        int model_in_h       = modelparams.nmodelinputheight_[0];
        int model_in_w       = modelparams.nmodelinputweith_[0];

        letterbox_t letter_box;
        memset(&letter_box, 0, sizeof(letterbox_t));
        convert_image_with_letterbox(model_in_w, model_in_h, image_in_w, image_in_h, &letter_box);

        post_process(modelparams, outputs, &letter_box, conf_threshold, nms_threshold, proposals);

    }

    ////家具检测----yolov8
    void yolov8_indoor_post_process(EcoRknnModelParams& modelparams, rknn_output outputs[3],  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_){
        /////在这里把我传进来的数据转成rknn_app_context_t、letterbox_t、object_detect_result_list等形式后调用

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold  = iou_out_Threshold_of_each_[0][0];
        conf_threshold = 0.25;

        int model_in_h       = modelparams.nmodelinputheight_[0];
        int model_in_w       = modelparams.nmodelinputweith_[0];

        letterbox_t letter_box;
        memset(&letter_box, 0, sizeof(letterbox_t));
        convert_image_with_letterbox(model_in_w, model_in_h, image_in_w, image_in_h, &letter_box);

        indoor_yolov8_post_process(modelparams, outputs, &letter_box, conf_threshold, nms_threshold, proposals);

    }

    ////家具检测----yolov8
    void yolov8_indoor_prop_post_process(EcoRknnModelParams& modelparams, rknn_output outputs[3],  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_){
        /////在这里把我传进来的数据转成rknn_app_context_t、letterbox_t、object_detect_result_list等形式后调用

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold  = iou_out_Threshold_of_each_[0][0];
        conf_threshold = 0.05;

        int model_in_h       = modelparams.nmodelinputheight_[0];
        int model_in_w       = modelparams.nmodelinputweith_[0];

        letterbox_t letter_box;
        memset(&letter_box, 0, sizeof(letterbox_t));
        convert_image_with_letterbox(model_in_w, model_in_h, image_in_w, image_in_h, &letter_box);

        indoor_prop_yolov8_post_process(modelparams, outputs, &letter_box, conf_threshold, nms_threshold, proposals);
    }

    void yolov8_indoor_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_, int want_float){

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold  = iou_out_Threshold_of_each_[0][0];
        conf_threshold = 0.25;

        int model_in_h       = modelparams.nmodelinputheight_[0];
        int model_in_w       = modelparams.nmodelinputweith_[0];

        letterbox_t letter_box;
        memset(&letter_box, 0, sizeof(letterbox_t));
        convert_image_with_letterbox(model_in_w, model_in_h, image_in_w, image_in_h, &letter_box);

        indoor_yolov8_post_process_zero_copy(modelparams, outputs, &letter_box, conf_threshold, nms_threshold, proposals, want_float);

    }

    // 零拷贝
    void yolov8_indoor_prop_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_,
                        int want_float){

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold  = iou_out_Threshold_of_each_[0][0];
        conf_threshold = 0.05;

        int model_in_h       = modelparams.nmodelinputheight_[0];
        int model_in_w       = modelparams.nmodelinputweith_[0];

        letterbox_t letter_box;
        memset(&letter_box, 0, sizeof(letterbox_t));
        convert_image_with_letterbox(model_in_w, model_in_h, image_in_w, image_in_h, &letter_box);

        indoor_prop_yolov8_post_process_zero_copy(modelparams, outputs, &letter_box, conf_threshold, nms_threshold, proposals, want_float);
    }

    void convert_image_with_letterbox(int model_in_w, int model_in_h, int image_in_w, int image_in_h, letterbox_t* letterbox)
    {
        int padding_w = 0;
        int padding_h = 0;
        int _left_offset = 0;
        int _top_offset = 0;
        float scale = 1.0;

        int dst_w = model_in_w;
        int dst_h = model_in_h;
        int src_w = image_in_w;
        int src_h = image_in_h;
        int resize_w = dst_w;
        int resize_h = dst_h;

        image_rect_t dst_box;
        dst_box.left = 0;
        dst_box.top = 0;
        dst_box.right = model_in_w- 1;
        dst_box.bottom = model_in_h - 1;

        ////这里的dst_w是模型输入的宽，src_w是原图的宽
        float _scale_w = (float)dst_w / src_w;
        float _scale_h = (float)dst_h / src_h;
        if(_scale_w < _scale_h) {
            scale = _scale_w;
            resize_h = (int) src_h*scale;
        } else {
            scale = _scale_h;
            resize_w = (int) src_w*scale;
        }

        // padding
        padding_h = dst_h - resize_h;
        padding_w = dst_w - resize_w;
        // center
        if (_scale_w < _scale_h) {
            dst_box.top = padding_h / 2;
            if (dst_box.top % 2 != 0) {
                dst_box.top -= dst_box.top % 2;
                if (dst_box.top < 0) {
                    dst_box.top = 0;
                }
            }
            dst_box.bottom = dst_box.top + resize_h - 1;
            _top_offset = dst_box.top;
        } else {
            dst_box.left = padding_w / 2;
            if (dst_box.left % 2 != 0) {
                dst_box.left -= dst_box.left % 2;
                if (dst_box.left < 0) {
                    dst_box.left = 0;
                }
            }
            dst_box.right = dst_box.left + resize_w - 1;
            _left_offset = dst_box.left;
        }
        //set offset and scale
        if(letterbox != NULL){
            letterbox->scale = scale;        ///原来
            letterbox->x_pad = _left_offset;
            letterbox->y_pad = _top_offset;
        }

    }



    static void compute_dfl(float* tensor, int dfl_len, float* box){
        for (int b = 0; b < 4; b ++)
        {
            float exp_t[dfl_len];
            float exp_sum = 0;
            float acc_sum = 0;
            for (int i = 0; i < dfl_len; i++){
                exp_t[i] = exp(tensor[i+b*dfl_len]);
                exp_sum += exp_t[i];
            }
            
            for (int i=0; i< dfl_len; i++){
                acc_sum += exp_t[i]/exp_sum * i;
            }
            box[b] = acc_sum;
        }
    }


    static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
    {
        float key;
        int key_index;
        int low  = left;
        int high = right;
        if (left < right)
        {
            key_index = indices[left];
            key = input[left];
            while (low < high)
            {
                while (low < high && input[high] <= key)
                {
                    high--;
                }
                input[low]   = input[high];
                indices[low] = indices[high];
                while (low < high && input[low] >= key)
                {
                    low++;
                }
                input[high]   = input[low];
                indices[high] = indices[low];
            }
            input[low]   = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }


    static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                        int8_t *score_tensor, int32_t score_zp, float score_scale,
                        int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int>   &classId,
                        float threshold,
                        int class_number)
    {
        int validCount = 0;
        int grid_len              = grid_h * grid_w;
        float unsig_threshold     = unsigmoid(threshold);
        int8_t score_thres_i8     = qnt_f32_to_affine(unsig_threshold, score_zp,     score_scale);
        int8_t score_sum_thres_i8 = qnt_f32_to_affine(unsig_threshold, score_sum_zp, score_sum_scale);


        float x1, y1, x2, y2, w, h;

        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;

                // 通过 score sum 起到快速过滤的作用
                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < score_sum_thres_i8)
                    {
                        continue;
                    }
                }

                int8_t max_score = -score_zp;
                for (int c = 0; c < class_number; c++)
                {
                    if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                    {
                        max_score    = score_tensor[offset];
                        max_class_id = c;
                    }
                    offset += grid_len;
                }

                // compute box
                if (max_score > score_thres_i8)
                {
                    offset = i* grid_w + j;
                    float box[4];
                    float before_dfl[dfl_len*4];
                    for (int k = 0; k < dfl_len * 4; k ++)
                    {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    x1 = (-box[0] + j + 0.5) * stride;
                    y1 = (-box[1] + i + 0.5) * stride;
                    x2 =  (box[2] + j + 0.5) * stride;
                    y2 =  (box[3] + i + 0.5) * stride;
                    w  = x2 - x1;
                    h  = y2 - y1;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    objProbs.push_back(sigmoid(deqnt_affine_to_f32(max_score, score_zp, score_scale)));
                    classId.push_back(max_class_id);
                    validCount ++;
                }
            }
        }
        return validCount;
    }



    static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                            int grid_h, int grid_w, int stride, int dfl_len,
                            std::vector<float> &boxes, 
                            std::vector<float> &objProbs, 
                            std::vector<int> &classId, 
                            float threshold,
                            int class_number)
    {
        int validCount = 0;
        int grid_len = grid_h * grid_w;


        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i* grid_w + j;
                int max_class_id = -1;

                // 通过 score sum 起到快速过滤的作用
                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < threshold)
                    {
                        continue;
                    }
                }

                float max_score = 0;
                for (int c= 0; c< class_number; c++)
                {
                    if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                    {
                        max_score = score_tensor[offset];
                        max_class_id = c;
                    }
                    offset += grid_len;
                }

                // compute box
                if (max_score> threshold)
                {
                    offset = i* grid_w + j;
                    float box[4];
                    float before_dfl[dfl_len*4];
                    for (int k=0; k< dfl_len*4; k++)
                    {
                        before_dfl[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1,y1,x2,y2,w,h;
                    x1 = (-box[0] + j + 0.5) * stride;
                    y1 = (-box[1] + i + 0.5) * stride;
                    x2 =  (box[2] + j + 0.5) * stride;
                    y2 =  (box[3] + i + 0.5) * stride;
                    w  = x2 - x1;
                    h  = y2 - y1;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);

                    objProbs.push_back(max_score);
                    classId.push_back(max_class_id);
                    validCount ++;
                }
            }
        }
        return validCount;
    }



    static int multitask_model_post_process(std::vector<detect_result_t>& proposals, EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, 
            int modelSwitch, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_)
    {
        // int idx = 0;
        int start_idx   = 0;
        int start_label = 0;

        int head_size = POSTPROCESSLIST.size();
        if(conf_out_threshold_of_each_.size() != head_size || iou_out_Threshold_of_each_.size() != head_size)
        {
            std::cout<<"ERROR! ERROR! The size of the Multitask_model head does not match the threshold size"<<std::endl;
        }

        for(int i = 0; i < POSTPROCESSLIST.size(); i++)
        {
            // // 根据输出长度先分配内存，然后送入对应的func中进行后处理操作
            // if(modelSwitch % 2 == 0){
            //     std::string head_string(POSTPROCESSLIST[i].head_name);
            //     if(head_string == "ground_petsshit")
            //     {
            //         start_idx   += POSTPROCESSLIST[i].output_struct_number;
            //         start_label += POSTPROCESSLIST[i].threshold_c;
            //         continue;
            //     }
            // }
            
            int result = POSTPROCESSLIST[i].func(modelparams, (outputs + start_idx), letter_box, conf_out_threshold_of_each_[i], iou_out_Threshold_of_each_[i], POSTPROCESSLIST[i].threshold_c, proposals, start_idx, start_label);
  
            start_idx   += POSTPROCESSLIST[i].output_struct_number;                    
            start_label += POSTPROCESSLIST[i].threshold_c;

        }

        return 0;
    }


    int yolov8_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, 
                            std::vector<float> conf_threshold, std::vector<float> nms_threshold, float class_number, std::vector<detect_result_t>& proposals,
                            int start_idx, int start_label)
    {

        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int>   classId;
        int validCount = 0;
        int stride     = 0;
        int grid_h     = 0;
        int grid_w     = 0;
        int model_in_w = modelparams.nmodelinputweith_[0];
        int model_in_h = modelparams.nmodelinputheight_[0];

        // default 3 branch
        int dfl_len = modelparams.nmodeloutputchannel_[0] / 4;
        int output_per_branch = 2;
        for (int i = 0; i < 3; i++)
        {
            void *score_sum = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0;
            if (output_per_branch == 3){
                score_sum       = outputs[i * output_per_branch + 2].buf;
                score_sum_zp    = modelparams.out_zps[i * output_per_branch + 2];
                score_sum_scale = modelparams.out_scales[i * output_per_branch + 2];
            }
            int box_idx   = i * output_per_branch;
            int score_idx = i * output_per_branch + 1;

            grid_h = modelparams.nmodeloutputheight_[box_idx];
            grid_w = modelparams.nmodeloutputweith_[box_idx];
            stride = model_in_h / grid_h;

            if (!outputs[0].want_float)
            {
                validCount += process_i8((int8_t *)outputs[box_idx].buf,   modelparams.out_zps[start_idx + box_idx], modelparams.out_scales[start_idx + box_idx],
                                        (int8_t *)outputs[score_idx].buf, modelparams.out_zps[start_idx + score_idx], modelparams.out_scales[start_idx + score_idx],
                                        (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                        grid_h, grid_w, stride, dfl_len, 
                                        filterBoxes, objProbs, classId, conf_threshold[0], int(class_number));
            }
            else
            {
                validCount += process_fp32((float *)outputs[box_idx].buf, (float *)outputs[score_idx].buf, (float *)score_sum,
                                        grid_h, grid_w, stride, dfl_len, 
                                        filterBoxes, objProbs, classId, conf_threshold[0], int(class_number));
            }
        }

        // no object detect
        if (validCount <= 0)
        {
            return 0;
        }
        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i)
        {
            indexArray.push_back(i);
        }
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));

        for (auto c : class_set)
        {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold[0]);
        }

        /* box valid detect target */
        for (int i = 0; i < validCount; ++i)
        {
            if (indexArray[i] == -1 || proposals.size() >= OBJ_NUMB_MAX_SIZE)
            {
                continue;
            }
            int   n  = indexArray[i];
            ////这两句主要为了应对纯resize的情况的
            float x1 = filterBoxes[n * 4 + 0];
            float y1 = filterBoxes[n * 4 + 1];
            float x2 = x1 + filterBoxes[n * 4 + 2];
            float y2 = y1 + filterBoxes[n * 4 + 3];
            int   id = classId[n];
            float obj_conf = objProbs[i];

            detect_result_t outputresult;
            BOX_PROP prop;
            outputresult.box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            outputresult.box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            outputresult.box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
            outputresult.box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);

            prop.condidence         = obj_conf;
            prop.name               = start_label + id;
            outputresult.prop.push_back(prop);
            outputresult.issure     = true;

            proposals.push_back(outputresult);

        }

        return 0;
    }

    // 零拷贝后处理
    // 修改rknn_output *_outputs为rknn_tensor_mem **_outputs，添加want_float参数
    // _outputs[].buf改为_outputs[]->virt_addr
    int yolov8_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, letterbox_t *letter_box, 
                            std::vector<float> conf_threshold, std::vector<float> nms_threshold, float class_number, std::vector<detect_result_t>& proposals,
                            int want_float)
    {

        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int>   classId;
        int validCount = 0;
        int stride     = 0;
        int grid_h     = 0;
        int grid_w     = 0;
        int model_in_w = modelparams.nmodelinputweith_[0];
        int model_in_h = modelparams.nmodelinputheight_[0];

        // default 3 branch
        int dfl_len = modelparams.nmodeloutputchannel_[0] / 4;
        int output_per_branch = 2;
        for (int i = 0; i < 2; i++)
        {
            void *score_sum = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0;
            if (output_per_branch == 3){
                score_sum       = outputs[i * output_per_branch + 2]->virt_addr;
                score_sum_zp    = modelparams.out_zps[i * output_per_branch + 2];
                score_sum_scale = modelparams.out_scales[i * output_per_branch + 2];
            }
            int box_idx   = i * output_per_branch;
            int score_idx = i * output_per_branch + 1;

            grid_h = modelparams.nmodeloutputheight_[box_idx];
            grid_w = modelparams.nmodeloutputweith_[box_idx];
            stride = model_in_h / grid_h;

            if (!want_float)
            {
                validCount += process_i8((int8_t *)outputs[box_idx]->virt_addr,   modelparams.out_zps[box_idx], modelparams.out_scales[box_idx],
                                        (int8_t *)outputs[score_idx]->virt_addr, modelparams.out_zps[score_idx], modelparams.out_scales[score_idx],
                                        (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                        grid_h, grid_w, stride, dfl_len, 
                                        filterBoxes, objProbs, classId, conf_threshold[0], int(class_number));
            }
            else
            {
                validCount += process_fp32((float *)outputs[box_idx]->virt_addr, (float *)outputs[score_idx]->virt_addr, (float *)score_sum,
                                        grid_h, grid_w, stride, dfl_len, 
                                        filterBoxes, objProbs, classId, conf_threshold[0], int(class_number));
            }
        }

        // no object detect
        if (validCount <= 0)
        {
            return 0;
        }
        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i)
        {
            indexArray.push_back(i);
        }
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));

        for (auto c : class_set)
        {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold[0]);
        }

        /* box valid detect target */
        for (int i = 0; i < validCount; ++i)
        {
            if (indexArray[i] == -1 || proposals.size() >= OBJ_NUMB_MAX_SIZE)
            {
                continue;
            }
            int   n  = indexArray[i];
            ////这两句主要为了应对纯resize的情况的
            float x1 = filterBoxes[n * 4 + 0];
            float y1 = filterBoxes[n * 4 + 1];
            float x2 = x1 + filterBoxes[n * 4 + 2];
            float y2 = y1 + filterBoxes[n * 4 + 3];
            int   id = classId[n];
            float obj_conf = objProbs[i];

            detect_result_t outputresult;
            BOX_PROP prop;
            outputresult.box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            outputresult.box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            outputresult.box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
            outputresult.box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);

            prop.condidence         = obj_conf;
            prop.name               = id;
            outputresult.prop.push_back(prop);
            outputresult.issure     = true;

            proposals.push_back(outputresult);

        }

        return 0;
    }



    static int group_points2box(int8_t *cls_pred, int32_t cls_zp, float cls_scale,
                                int8_t *reg_pred, int32_t reg_zp, float reg_scale,
                                int start_row, int start_col,
                                int grid_h, int grid_w, int stride, 
                                float ratio_h, float ratio_w,
                                std::vector<float> &boxes, 
                                std::vector<float> &objProbs, 
                                int8_t cls_thres_i8, float line_box_threshold, float line_air_threshold_float)                        
    {
        int grid_len       = grid_h * grid_w;
        int8_t max_score   = -128;
        float dequan_score = 0.;
        float xmin         = 9999.f;
        float ymin         = 9999.f;
        float xmax         = 0.f;
        float ymax         = 0.f;
        int label          = -1;
        for (int row = start_row; row < start_row + stride; row++)
        {   
            for (int col = start_col; col < start_col + stride; col++)
            {  
                int offset   = row * grid_w + col;
                int8_t score = cls_pred[offset];
                if (score > cls_thres_i8)
                {
                    int8_t x_set = reg_pred[offset];
                    int8_t y_set = reg_pred[grid_len + offset];
                    float x = (col + deqnt_affine_to_f32(x_set, reg_zp, reg_scale)) * ratio_w;
                    float y = (row + deqnt_affine_to_f32(y_set, reg_zp, reg_scale)) * ratio_h;
                    if (x < xmin){
                        xmin=x;
                    }
                    if (y < ymin){
                        ymin=y;
                    }
                    if (x > xmax){
                        xmax=x;
                    }
                    if (y > ymax){
                        ymax=y;
                    }
                    if (score > max_score){
                        max_score=score;
                    }
                    label = 1;
                }
            }
        }
        dequan_score = sigmoid(deqnt_affine_to_f32(max_score, cls_zp, cls_scale));
        if (label==1 && dequan_score > line_box_threshold && ymax > line_air_threshold_float)
        {
            float w,h;
            w = xmax - xmin;
            h = ymax - ymin;
            boxes.push_back(xmin);
            boxes.push_back(ymin);
            boxes.push_back(w);
            boxes.push_back(h);
            objProbs.push_back(dequan_score);
            return 1;
        }
        return 0;
    }
    



    static int process_i8(int8_t *cls_pred, int32_t cls_zp, float cls_scale,
                        int8_t *reg_pred, int32_t reg_zp, float reg_scale,
                        int grid_h, int grid_w, int stride,
                        float ratio_h, float ratio_w,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        float threshold, float line_box_threshold, float line_air_threshold_float)
    {
        int   validCount      = 0;
        float unsig_threshold = unsigmoid(threshold);   // 阈值量化
        int8_t cls_thres_i8   = qnt_f32_to_affine(unsig_threshold, cls_zp, cls_scale);
        for (int start_row = 0; start_row < grid_h; start_row += stride)
        {
            for (int start_col = 0; start_col < grid_w; start_col += stride)
            {
                validCount += group_points2box(cls_pred, cls_zp, cls_scale, reg_pred, reg_zp, reg_scale,
                                            start_row, start_col,
                                            grid_h, grid_w, stride, 
                                            ratio_h, ratio_w, boxes, objProbs, cls_thres_i8, line_box_threshold, line_air_threshold_float);
            }
        }
        return validCount;
    }


    int line_post_process(EcoRknnModelParams& modelparams, rknn_output *outputs, letterbox_t *letter_box, 
                        std::vector<float> conf_threshold, std::vector<float> line_box_threshold, float line_air_threshold_float, std::vector<detect_result_t>& od_results, 
                        int start_idx, int start_label)
    {
        int cls_idx    = 1;
        int reg_idx    = 0;
        int validCount = 0;
        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        filterBoxes.clear();
        objProbs.clear();

        int grid_h     = modelparams.nmodeloutputheight_[0];
        int grid_w     = modelparams.nmodeloutputweith_[0];
        int stride     = modelparams.nmodelinputheight_[0] / grid_h;
        float ratio_w  = modelparams.nmodelinputweith_[0]  / grid_w;
        float ratio_h  = modelparams.nmodelinputheight_[0] / grid_h;
        int model_in_w = modelparams.nmodelinputweith_[0];
        int model_in_h = modelparams.nmodelinputheight_[0];
        float line_air_threshold_float_ratio = line_air_threshold_float / ratio_h;

        if (!outputs[0].want_float)
        {
            validCount += process_i8((int8_t *)outputs[cls_idx].buf, modelparams.out_zps[start_idx + cls_idx], modelparams.out_scales[start_idx + cls_idx],
                                    (int8_t *)outputs[reg_idx].buf, modelparams.out_zps[start_idx + reg_idx], modelparams.out_scales[start_idx + reg_idx],
                                    grid_h, grid_w, stride, ratio_h, ratio_w,
                                    filterBoxes, objProbs, conf_threshold[0], line_box_threshold[0], line_air_threshold_float_ratio);
        }
        // no object detect
        if (validCount <= 0)
        {
            return 0;
        }


        /* box valid detect target */
        for (int i = 0; i < validCount; ++i)
        {
            ////下面两句是为了应对有上下padding的时候的
            float x1 = filterBoxes[i * 4 + 0];
            float y1 = filterBoxes[i * 4 + 1];
            float x2 = x1 + filterBoxes[i * 4 + 2];
            float y2 = y1 + filterBoxes[i * 4 + 3];
            int   id = 0;
            float obj_conf = objProbs[i];

            detect_result_t outputresult;
            outputresult.prop.clear();
            BOX_PROP prop;

            outputresult.box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            outputresult.box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            outputresult.box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
            outputresult.box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);

            prop.condidence         = obj_conf;
            prop.name               = start_label + id;
            outputresult.prop.push_back(prop);
            outputresult.issure = true;

            od_results.push_back(outputresult);

        }
        return 0;
    }

    /////多任务模型---yolov8--line
    void Multitask_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch)
    {
        letterbox_t letter_box;
        letter_box.scale = (float)modelparams.nmodelinputheight_[0] / (float)image_in_h;
        letter_box.x_pad = 0;
        letter_box.y_pad = 0;

        multitask_model_post_process(proposals, modelparams, outputs, &letter_box, modelSwitch, conf_out_threshold_of_each_, iou_out_Threshold_of_each_);
    }

    // 零拷贝 仅保留检测头的后处理
    void multitask_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float)
    {
        letterbox_t letter_box;
        letter_box.scale = (float)modelparams.nmodelinputheight_[0] / (float)image_in_h;
        letter_box.x_pad = 0;
        letter_box.y_pad = 0;

        yolov8_post_process_zero_copy(modelparams, outputs, &letter_box, conf_out_threshold_of_each_[0], iou_out_Threshold_of_each_[0], 6, proposals, want_float);
    }

    void freespace_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs,  cv::Mat &mask,  std::vector<EcoKeyPoint>& maskdata,
                        int image_in_w, int image_in_h, float clsthreshold_, float iouThreshold_, std::vector<std::vector<float>> out_threshold_of_all, int& cm_distance)
    {
        /////把outputs中的相关内容转移到rknn_app_ctx
        int model_width        = (modelparams.nmodelinputweith_)[0];
        int model_height       = (modelparams.nmodelinputheight_)[0];
        int num_class_         = modelparams.nmodeloutputchannel_[0]; 
        int outputdata_size    = modelparams.io_num.n_output;
        float score_threshold_ = out_threshold_of_all[0][0];
        float nms_threshold_   = iouThreshold_;

        std::vector<BBox> result_boxes;
        result_boxes.clear();
        // gfl_post_process(result_boxes, outputs , num_class_, outputdata_size, model_height, model_width, image_in_h, image_in_w, score_threshold_, nms_threshold_, &modelparams.nmodeloutputweith_, &modelparams.nmodeloutputheight_);
       
        // 分割输出 
        int seg_outheight      = (modelparams.nmodeloutputchannel_)[modelparams.io_num.n_output-1];
        int seg_outweith       = (modelparams.nmodeloutputheight_)[modelparams.io_num.n_output-1];
        int seg_outchannel     = (modelparams.nmodeloutputweith_)[modelparams.io_num.n_output-1];
        
        // uint8_t* seg_out_data  = (uint8_t*)(outputs[6].buf); // float转uint8_t,128x128x7
        uint8_t* seg_out_data  = (uint8_t*)(outputs[0].buf); // float转uint8_t,128x128x7

        result_counter p; // 边缘提点结果
        p.ps.clear();
        p.id = 0;
        find_classes_contours(mask, p, result_boxes, seg_out_data, seg_outheight, seg_outweith, model_height, 
                model_width, image_in_h, image_in_w, cm_distance); 

        ///把得到的p中所有点坐标一一存放到分割中的最终处理结果中
        EcoKeyPoint keypoint_single;

        for(int i=0; i<p.ps.size(); i += 2)
        {
            keypoint_single.bistrue     = true;
            keypoint_single.mappos      = cv::Point2f(p.ps[i].x, p.ps[i].y);
            keypoint_single.id          = 0;
            keypoint_single.inlabel     = int(p.ps[i].label);
            keypoint_single.fconfidence = p.ps[i].confidence;
            keypoint_single.total_num_curve = 1; /////代表单张图总的曲线数量
            maskdata.push_back(keypoint_single);
        }

    }

    // 零拷贝
    void freespace_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  cv::Mat &mask,  std::vector<EcoKeyPoint>& maskdata,
                        int image_in_w, int image_in_h, float clsthreshold_, float iouThreshold_, std::vector<std::vector<float>> out_threshold_of_all, int& cm_distance)
    {
        /////把outputs中的相关内容转移到rknn_app_ctx
        int model_width        = (modelparams.nmodelinputweith_)[0];
        int model_height       = (modelparams.nmodelinputheight_)[0];
        int num_class_         = modelparams.nmodeloutputchannel_[0]; 
        int outputdata_size    = modelparams.io_num.n_output;
        float score_threshold_ = out_threshold_of_all[0][0];
        float nms_threshold_   = iouThreshold_;

        std::vector<BBox> result_boxes;
        result_boxes.clear();
        // gfl_post_process(result_boxes, outputs , num_class_, outputdata_size, model_height, model_width, image_in_h, image_in_w, score_threshold_, nms_threshold_, &modelparams.nmodeloutputweith_, &modelparams.nmodeloutputheight_);
       
        // 分割输出 
        int seg_outheight      = (modelparams.nmodeloutputchannel_)[modelparams.io_num.n_output-1];
        int seg_outweith       = (modelparams.nmodeloutputheight_)[modelparams.io_num.n_output-1];
        int seg_outchannel     = (modelparams.nmodeloutputweith_)[modelparams.io_num.n_output-1];

        result_counter p; // 边缘提点结果
        p.ps.clear();
        p.id = 0;
        int32_t zp = modelparams.out_zps[0];
        float scale = modelparams.out_scales[0];

        // 零拷贝 模型结果修改为int8传入
        find_classes_contours_zero_copy(mask, p, result_boxes, (int8_t *)outputs[0]->virt_addr, seg_outheight, seg_outweith, model_height, 
                model_width, image_in_h, image_in_w, cm_distance, zp, scale); 
        
        // 零拷贝 优化后处理
        // find_classes_contours_fast_zero_copy(mask, p, result_boxes, (int8_t *)outputs[0]->virt_addr, seg_outheight, seg_outweith, model_height, 
        //         model_width, image_in_h, image_in_w, cm_distance); 

        ///把得到的p中所有点坐标一一存放到分割中的最终处理结果中
        EcoKeyPoint keypoint_single;

        for(int i=0; i<p.ps.size(); i += 1)
        {
            keypoint_single.bistrue     = true;
            keypoint_single.mappos      = cv::Point2f(p.ps[i].x, p.ps[i].y);
            keypoint_single.id          = 0;
            keypoint_single.inlabel     = int(p.ps[i].label);
            keypoint_single.fconfidence = p.ps[i].confidence;
            keypoint_single.total_num_curve = 1; /////代表单张图总的曲线数量
            maskdata.push_back(keypoint_single);
        }

    }




    void yolox_peopledet_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_)
    {

        float conf_threshold = conf_out_threshold_of_each_[0][0];   ////
        float nms_threshold = iou_out_Threshold_of_each_[0][0];


        int model_in_h = modelparams.nmodelinputheight_[0];
        int model_in_w = modelparams.nmodelinputweith_[0];

        float ratio_ = MIN(float(model_in_w)/image_in_w, float(model_in_h)/image_in_h);
        YoloxModel model;
        std::vector<BBox> results = model.yolox_postprocess(outputs, image_in_h, image_in_w, conf_threshold, nms_threshold, ratio_);

        std::vector<BodyBox> bboxes;
        for (size_t i = 0; i < int(results.size()); i++)
        {
            BodyBox tmp;
            bboxes.push_back(BodyBox{ results[i].xmin, 
                        results[i].ymin , 
                        results[i].xmax, 
                        results[i].ymax, 
                        float(results[i].label), results[i].score, 0.f, 0.f});
        }

        model.humandet_boxes_filter(bboxes, 0.7);

        std::vector<BodyBox> newbodybox;
        for(int i=0; i < bboxes.size(); i++){

            if(bboxes[i].label != -1){
                //printf("pushback\n");
                newbodybox.push_back(bboxes[i]);
            }
        }


        /////在这里把模型出来的名字转换为
        for (int i = 0; i < newbodybox.size(); i++)
        {
            detect_result_t result_a;
            BOX_PROP result_box_prop;
            result_a.box.left = newbodybox[i].xmin;
            result_a.box.top = newbodybox[i].ymin;
            result_a.box.right = newbodybox[i].xmax;
            result_a.box.bottom = newbodybox[i].ymax;
            result_box_prop.condidence = newbodybox[i].score;
            ////这里的cls_id是和txt中的name索引对应起来--name最后转成inlabel
            result_box_prop.name = int(newbodybox[i].label);    
            result_a.prop.push_back(result_box_prop);
            result_a.issure = true;
            proposals.push_back(result_a);
        }
    }

    // 人形 零拷贝
    void yolox_peopledet_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>> conf_out_threshold_of_each_, std::vector<std::vector<float>> iou_out_Threshold_of_each_)
    {

        float conf_threshold = conf_out_threshold_of_each_[0][0];   ////
        float nms_threshold = iou_out_Threshold_of_each_[0][0];


        int model_in_h = modelparams.nmodelinputheight_[0];
        int model_in_w = modelparams.nmodelinputweith_[0];

        float ratio_ = MIN(float(model_in_w)/image_in_w, float(model_in_h)/image_in_h);
        YoloxModel model;
        // 零拷贝 模型结果修改为int8传入
        // 增加modelparams参数传入，用于获取zp和scale，将模型输出转为float
        std::vector<BBox> results = model.yolox_postprocess_zero_copy(modelparams, (int8_t *)outputs[0]->virt_addr, image_in_h, image_in_w, conf_threshold, nms_threshold, ratio_);

        std::vector<BodyBox> bboxes;
        for (size_t i = 0; i < int(results.size()); i++)
        {
            BodyBox tmp;
            bboxes.push_back(BodyBox{ results[i].xmin, 
                        results[i].ymin , 
                        results[i].xmax, 
                        results[i].ymax, 
                        float(results[i].label), results[i].score, 0.f, 0.f});
        }

        model.humandet_boxes_filter(bboxes, 0.7);

        std::vector<BodyBox> newbodybox;
        for(int i=0; i < bboxes.size(); i++){

            if(bboxes[i].label != -1){
                //printf("pushback\n");
                newbodybox.push_back(bboxes[i]);
            }
        }


        /////在这里把模型出来的名字转换为
        for (int i = 0; i < newbodybox.size(); i++)
        {
            detect_result_t result_a;
            BOX_PROP result_box_prop;
            result_a.box.left = newbodybox[i].xmin;
            result_a.box.top = newbodybox[i].ymin;
            result_a.box.right = newbodybox[i].xmax;
            result_a.box.bottom = newbodybox[i].ymax;
            result_box_prop.condidence = newbodybox[i].score;
            ////这里的cls_id是和txt中的name索引对应起来--name最后转成inlabel
            result_box_prop.name = int(newbodybox[i].label);    
            result_a.prop.push_back(result_box_prop);
            result_a.issure = true;
            proposals.push_back(result_a);
        }
    }


    void yolov5_PM_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch){      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h - 208);

        pm_post_process(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals);

    }

    void yolov5_PM_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float){      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h - 208);

        pm_post_process_zero_copy(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void yolov5_Liquid_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float){      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h - 208);

        liquid_post_process_zero_copy_v5(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void yolov8_Liquid_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float){      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h - 520);

        liquid_post_process_zero_copy(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void yolov5_lint_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float){      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h - 208);

        lint_post_process_zero_copy(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void yolov5_drystain_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float){      
        
        std::vector<float> conf_threshold = conf_out_threshold_of_each_[0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        // std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h - 208);

        drystain_post_process_zero_copy(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void yolov8_irstain_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch,
                        int want_float){

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        conf_threshold = conf_threshold * 0.7;
        std::cout << "conf_threshold__ = " << conf_threshold << std::endl;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h);

        irstain_post_process_zero_copy_v8(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void dirt_det_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int modelSwitch)
    {      

        // std::cout<<"start dirt_det_post_process"<<std::endl;
        float cls_threshold = conf_out_threshold_of_each_[0][0];

        std::vector<BBox> roi_boxes;
        roi_boxes.clear();

        int stepx = image_in_w/8;
        int stepy = image_in_h/4; // 45

        for (size_t row = 0; row < image_in_h; row+=stepy) //row = imgh - 180
        {
            for (size_t col = 0; col < image_in_w; col+=stepx)
            {
                roi_boxes.push_back(BBox{ float(col), float(row), float(col+ stepx), float(row + stepy), 0.f, 0.f});
            }
        }
        // std::cout<<"roi_boxes.size(): "<<roi_boxes.size()<<std::endl;

        float* scores =  (float*)outputs[0].buf;
        int label = 0;
        float score;

        for (auto i = 0; i < 32; i++) {
            // std::cout<<"i: "<<i<<"-----scores[i]: "<<scores[i]<<std::endl;
            score = sigmoid(scores[i]);
            // std::cout<<"score: "<<score<<std::endl;
            detect_result_t outputresult;
            BOX_PROP prop;
            if(score > cls_threshold){
                outputresult.box.left   = (int)(roi_boxes[i].xmin);
                outputresult.box.top    = (int)(roi_boxes[i].ymin + 520);
                outputresult.box.right  = (int)(roi_boxes[i].xmax);
                outputresult.box.bottom = (int)(roi_boxes[i].ymax + 520);
                prop.condidence         = score;
                prop.name               = 0;
                outputresult.prop.push_back(prop);
                outputresult.issure     = true;
                proposals.push_back(outputresult);
            }
        }
        // std::cout<<"end dirt_det__post_process: "<<proposals.size()<<std::endl;
    }


    void wuzi_int8_det_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int modelSwitch)
    {      

        // std::cout<<"start dirt_det_post_process"<<std::endl;
        float cls_threshold = conf_out_threshold_of_each_[0][0];
        int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(cls_threshold), modelparams.out_zps[3], modelparams.out_scales[3]);

        int grid_w = 8;
        int grid_h = 4;
        int step_x = image_in_w / grid_w;
        int step_y = image_in_h / grid_h;
        int8_t *input = (int8_t *)outputs[3].buf;
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int8_t score = input[i*grid_w + j];
                detect_result_t outputresult;
                BOX_PROP prop;
                if (score >= thres_i8)
                {
                    float score_ = sigmoid(deqnt_affine_to_f32(score, modelparams.out_zps[3], modelparams.out_scales[3]));
                    outputresult.box.left   = (int)(j * step_x);
                    outputresult.box.top    = (int)(i * step_y + 520);
                    outputresult.box.right  = (int)((j + 1) * step_x);
                    outputresult.box.bottom = (int)((i + 1) * step_y + 520);
                    prop.condidence         = score_;
                    prop.name               = 3;
                    outputresult.prop.push_back(prop);
                    outputresult.issure     = true;
                    proposals.push_back(outputresult);
                    // printf("%d %d,%f %f %f %f %f\n",i,j,score_, x1,y1,x2,y2);
                }
            }
        }
    }








    void softmax(float* array, int size) {
    // Find the maximum value in the array
        float max_val = array[0];
        for (int i = 1; i < size; i++) {
            if (array[i] > max_val) {
                max_val = array[i];
            }
        }

        // Subtract the maximum value from each element to avoid overflow
        for (int i = 0; i < size; i++) {
            array[i] -= max_val;
        }

        // Compute the exponentials and sum
        float sum = 0.0;
        for (int i = 0; i < size; i++) {
            array[i] = expf(array[i]);
            sum += array[i];
        }

        // Normalize the array by dividing each element by the sum
        for (int i = 0; i < size; i++) {
            array[i] /= sum;
        }
    }

    typedef struct {
        float value;
        int index;
    } element_t;

    void swap(element_t* a, element_t* b) {
        element_t temp = *a;
        *a = *b;
        *b = temp;
    }

    int partition(element_t arr[], int low, int high) {
        float pivot = arr[high].value;
        int i = low - 1;

        for (int j = low; j <= high - 1; j++) {
            if (arr[j].value >= pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }

        swap(&arr[i + 1], &arr[high]);
        return (i + 1);
    }


    void quick_sort(element_t arr[], int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quick_sort(arr, low, pi - 1);
            quick_sort(arr, pi + 1, high);
        }
    }


    void get_topk_with_indices(float arr[], int size, int k, std::vector<detect_result_t>& proposals) {

        // 创建元素数组，保存值和索引号
        // element_t* elements = (element_t*)malloc(size * sizeof(element_t));
        element_t elements[size];
        for (int i = 0; i < size; i++) 
        {
            elements[i].value = arr[i];
            elements[i].index = i;
        }
        detect_result_t outputresult;
        BOX_PROP        prop;

        // 对元素数组进行快速排序
        quick_sort(elements, 0, size - 1);

        // 获取前K个最大值和它们的索引号
        for (int i = 0; i < k; i++) {
            outputresult.box.left   = 100;
            outputresult.box.top    = 900;
            outputresult.box.right  = 300;
            outputresult.box.bottom = 940;
            prop.condidence         = elements[i].value;
            prop.name               = elements[i].index;
            outputresult.prop.push_back(prop);
            outputresult.issure     = true;
            proposals.push_back(outputresult);
        }
    }


    void bed_det_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int topk)
    {      

        // int n_elems_size = modelparams.nmodeloutputchannel_[0] * modelparams.nmodeloutputheight_[0] * modelparams.nmodeloutputweith_[0];
        int n_elems_size = modelparams.nmodeloutputheight_[0];
        softmax((float*)outputs[0].buf, n_elems_size);
        get_topk_with_indices((float*)outputs[0].buf, n_elems_size, topk, proposals);

    }

    void bed_det_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, int topk)
    {      

        // int n_elems_size = modelparams.nmodeloutputchannel_[0] * modelparams.nmodeloutputheight_[0] * modelparams.nmodeloutputweith_[0];
        int n_elems_size = modelparams.nmodeloutputheight_[0];
        float *outputs_float = new float[n_elems_size];
        int2float((int8_t *)outputs[0]->virt_addr, outputs_float, n_elems_size, modelparams.out_zps[0], modelparams.out_scales[0]);
        softmax(outputs_float, n_elems_size);
        get_topk_with_indices(outputs_float, n_elems_size, topk, proposals);
        delete[] outputs_float;
    }

    void int2float(int8_t *outputs, float *outputs_float, int size, int32_t zp, float scale)
    {
        for (int i = 0; i < size; i++) {
            outputs_float[i] = ((float)outputs[i] - (float)zp) * scale;
        }
    }

    void groundpoint_postprocess(EcoRknnModelParams& modelparams, rknn_output* outputs, std::vector<EcoKeyPoint>& maskdata,
                        int image_in_w, int image_in_h, float conf_threshold, int& cm_distance)
    {
        groundLine_det_post_process(modelparams, outputs, conf_threshold, image_in_h,  image_in_w, maskdata, cm_distance);
        groundSemantic_det_post_process(modelparams, outputs, conf_threshold, image_in_h,  image_in_w, maskdata, cm_distance);
        groundpoint_det_post_process(modelparams, outputs, conf_threshold, image_in_h,  image_in_w, maskdata, cm_distance);
    }

    // 零拷贝
    void groundpoint_postprocess_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs, std::vector<EcoKeyPoint>& maskdata,
                        int image_in_w, int image_in_h, float conf_threshold, int& cm_distance)
    {
        //接地线模型三个分支头：1.电线检测；2.地毯检测；3.无语义的通用接地点检测
        groundLine_det_post_process_zero_copy(modelparams, outputs, conf_threshold, image_in_h,  image_in_w, maskdata, cm_distance);
        groundSemantic_det_post_process_zero_copy(modelparams, outputs, conf_threshold, image_in_h,  image_in_w, maskdata, cm_distance);
        groundpoint_det_post_process_zero_copy(modelparams, outputs, conf_threshold, image_in_h,  image_in_w, maskdata, cm_distance);
    }


    void yolov5_IR_detect_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch)
    {      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        letterbox_t letter_box;
        letter_box.scale = 1.0;

        scale_t scale_wh;
        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h);

        infrared_post_process(modelparams, outputs, &letter_box, &scale_wh, conf_threshold, nms_threshold, proposals);

    }

    void yolov5_animal_detect_post_process(EcoRknnModelParams& modelparams, rknn_output* outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch)
    {      
        
        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        letterbox_t letter_box;
        letter_box.scale = 2.5;

        scale_t scale_wh;


        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h);

        animal_post_process(modelparams, outputs, &letter_box, &scale_wh, conf_threshold, nms_threshold, proposals);

    }

    void yolov5_animal_detect_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float)
    {      

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        letterbox_t letter_box;
        letter_box.scale = 2.5;

        scale_t scale_wh;


        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h);

        animal_post_process_zero_copy(modelparams, outputs, &letter_box, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void yolov11_animal_detect_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float)
    {      

        float conf_threshold = conf_out_threshold_of_each_[0][0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        letterbox_t letter_box;
        letter_box.scale = 2.5;

        scale_t scale_wh;


        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h);

        animal_yolov11_post_process_zero_copy(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals);

    }

    void yolov8_obstacle_detect_post_process_zero_copy(EcoRknnModelParams& modelparams, rknn_tensor_mem **outputs,  std::vector<detect_result_t> &proposals,
                        int image_in_w, int image_in_h, float detectThreshold_, std::vector<std::vector<float>>& conf_out_threshold_of_each_, std::vector<std::vector<float>>& iou_out_Threshold_of_each_, int modelSwitch, int want_float)
    {      

        std::vector<float> conf_threshold = conf_out_threshold_of_each_[0];
        float nms_threshold = iou_out_Threshold_of_each_[0][0];
        letterbox_t letter_box;
        letter_box.scale = 2.5;

        scale_t scale_wh;


        memset(&scale_wh, 0, sizeof(scale_t));
        scale_wh.scale_w = float(modelparams.nmodelinputweith_[0] ) / (image_in_w);
        scale_wh.scale_h = float(modelparams.nmodelinputheight_[0]) / (image_in_h);

        obstacle_yolov8_post_process_zero_copy(modelparams, outputs, &scale_wh, conf_threshold, nms_threshold, proposals, want_float);

    }

    void groundpoint2rug(EcoInstanceObjectSeg* ecoCamOutputResult, EcoSegInference* ecoGroundPointsseg_, 
    EcoSegInference* ecoFreespacetargetseg_, cv::Mat&  rug_mask)
    {
        // rug_mask 中地毯的值是  125
        // 地毯数据过滤
        cv::Point3f max_keypoint(cv::Point3f(0,0,0));
        cv::Point3f mix_keypoint(cv::Point3f(1000,1000,0));
        for(int kk = 0; kk < ecoGroundPointsseg_->getSegMasks()->maskdata.size(); kk++)
        {
            if(ecoFreespacetargetseg_->getSegMasks()->maskdata.size() > 0)
            {
                if(ecoGroundPointsseg_->getSegMasks()->maskdata[kk].bistrue)
                {
                    //  判断接地线在地毯边缘 则赋值接地线地毯语义
                    // if (brug_seg_remove(ecoGroundPointsseg_->getSegMasks()->maskdata[kk], ecoCamOutputResult->mask, max_keypoint, mix_keypoint, rug_mask))
                    // {
                    //     ecoCamOutputResult->maskdata.push_back(ecoGroundPointsseg_->getSegMasks()->maskdata[kk]);
                    // }
                    // 地毯点和接地点融合仅在bev空间进行
                    if (brug_seg_remove_bev(ecoGroundPointsseg_->getSegMasks()->maskdata[kk], ecoCamOutputResult->mask, max_keypoint, mix_keypoint, rug_mask))
                    {
                        ecoCamOutputResult->maskdata.push_back(ecoGroundPointsseg_->getSegMasks()->maskdata[kk]);
                    }
                }
            }
            else if(ecoGroundPointsseg_->getSegMasks()->maskdata[kk].fconfidence > 0)
            {
                ecoCamOutputResult->maskdata.push_back(ecoGroundPointsseg_->getSegMasks()->maskdata[kk]);
            }
        }

        for(int kk = 0; kk < ecoFreespacetargetseg_->getSegMasks()->maskdata.size(); kk++)
        {
            if(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].label == EM_OUT_TASSELS)
            {
                ecoCamOutputResult->maskdata.push_back(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk]);
                continue;
            }

            if(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].label == EM_OUT_CARPET && ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].bistrue)
            {
                if(max_keypoint.x - mix_keypoint.x >= 20 || ecoCamOutputResult->mask.at<uchar>(0, 0) == 0)
                {
                    if(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.x > (max_keypoint.x - 2))
                    {
                        ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].bistrue = false;
                        rug_mask.at<uchar>(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.x / 4, (0 - ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.y) / 4 + 40) = 200;
                        continue;
                    }
                }
                if(max_keypoint.x - mix_keypoint.x < 20 && ecoCamOutputResult->mask.at<uchar>(0, 0) == 50)
                {
                    if(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.x >= mix_keypoint.x + 40)
                    {
                        ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].bistrue = false;
                        rug_mask.at<uchar>(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.x / 4, (0 - ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.y) / 4 + 40) = 200;
                        continue;
                    }
                } 
                ecoCamOutputResult->maskdata.push_back(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk]);
            }
        }
    }

    void rug_multiFrameMatch(std::vector<EcoKeyPoint>& Freespace_maskdata, EcoAInterfaceDeebotStatus_t pose, std::deque<std::unordered_set<int>>& coords_w_queue, int max_match_frame, int num_match_frame, int grid_size)
    {
        std::vector<int> is_rug_idx;  // maskdata中有效地毯点的索引
        std::vector<int> is_rug_points(Freespace_maskdata.size(), 999);  // 999表示非地毯点，2表示地毯点，-2表示被消除的地毯点
        
        float radian = pose.Qz;
        std::unordered_set<int> coords_w_all;  // 世界坐标系下的地毯点及其冗余点

        // arr_coords_w 世界坐标系下的地毯点（不包含冗余），用于匹配历史帧数据
        int len_arr_coords_w = 1000;  
        int arr_coords_w[len_arr_coords_w];
        for (int i = 0; i < len_arr_coords_w; i++)
            arr_coords_w[i] = -99999;
        
        int min_x = 100000;
        int min_y = 100000;
        int max_x = -99999;
        int max_y = -99999;

        // arr_local 的 index 对 arr_coords_w_all 的点去重并 localization，value 保留 encoder 的结果
        int len_arr_local_x = 80;
        int len_arr_local_y = 100;
        int arr_local[len_arr_local_x][len_arr_local_y];  
        for (int i = 0; i < len_arr_local_x; ++i) 
            for (int j = 0; j < len_arr_local_y; ++j) 
                arr_local[i][j] = -99999;

        std::vector<std::pair<int, int>> coords_w_all_v;

        // 有效地毯points 转到世界坐标系（单位mm），并encoder: x + y * 1000
        for(int i = 0; i < Freespace_maskdata.size(); i++){
            if(Freespace_maskdata[i].inlabel == 2 && Freespace_maskdata[i].bistrue){

                float point_x = Freespace_maskdata[i].keypoint.x * 10 + 170;  // 机器坐标系，mm
                float point_y = Freespace_maskdata[i].keypoint.y * 10;

                float point_wx = std::round(point_x * std::cos(radian) - point_y * std::sin(radian) + pose.x);  // 世界坐标系下的栅格空间
                float point_wy = std::round(point_x * std::sin(radian) + point_y * std::cos(radian) + pose.y);
                int point_voxel_wx = point_wx / grid_size;
                int point_voxel_wy = point_wy / grid_size;
                
                // localization 范围
                if (point_voxel_wx < min_x)  
                    min_x = point_voxel_wx;
                if (point_voxel_wy < min_y)
                    min_y = point_voxel_wy;
                if (point_voxel_wx > max_x)
                    max_x = point_voxel_wx;
                if (point_voxel_wy > max_y) 
                    max_y = point_voxel_wy;

                arr_coords_w[is_rug_idx.size()] = point_voxel_wx + point_voxel_wy * 1000;  // 控制数组 arr_coords_w 的长度

                // // 地毯点及冗余点（用于放入队列，作为下一帧数据的历史帧）
                for (int dx = -1; dx <= 1; dx++)  
                    for (int dy = -1; dy <= 1; dy++)
                        coords_w_all_v.push_back({point_voxel_wx + dx, point_voxel_wy + dy});
                
                is_rug_points[i] = 2;
                is_rug_idx.push_back(i);
            }
        }


        // 将global栅格转至local栅格，利用 array 替换 unordered_set 的去重功能，并encoder
        for (int i = 0; i < coords_w_all_v.size(); i++){
                int local_wx = coords_w_all_v[i].first - min_x; 
                int local_wy = coords_w_all_v[i].second - min_y;

                if (local_wx >= 50){
                    std::cout << "!!!!! rugMatch local_wx: "  << local_wx <<std::endl;
                }
                if (local_wy >= 80){
                    std::cout << "!!!!! rugMatch local_wy: "  << local_wy <<std::endl;
                }

                arr_local[local_wx][local_wy] = coords_w_all_v[i].first + coords_w_all_v[i].second * 1000;
        }

        // 去重后的数组 insert 到 unordered_set
        coords_w_all.reserve(3000);  // 预先分配内存，加速insert
        int count_set = 0;
        for (int i = 0; i < len_arr_local_x; i++)
            for (int j = 0; j < len_arr_local_y; j++)
                if (arr_local[i][j] != -99999){
                    coords_w_all.insert(arr_local[i][j]);
                    count_set++;
                }
   
        // 进行多帧匹配
        if (coords_w_queue.size() == max_match_frame - 1){
            for (int i = 0; i < is_rug_idx.size(); i++){
                int dd_count = 0;
                for (const auto& coords_w_frame : coords_w_queue)
                    if (coords_w_frame.find(arr_coords_w[i]) != coords_w_frame.end()){

                        dd_count++;
                        if (dd_count >= num_match_frame -1)  // 提前结束匹配
                            break;
                    }
                if (dd_count < num_match_frame -1)
                    is_rug_points[is_rug_idx[i]] = -2;    
            }

            coords_w_queue.pop_front(); 
        }


        coords_w_queue.push_back(coords_w_all);

        // 目前消除掉的点 inlabel 改为-2，对外标签label未改动
        for(int i = 0; i < Freespace_maskdata.size(); i++)
            if(Freespace_maskdata[i].inlabel == 2 && Freespace_maskdata[i].bistrue)
                Freespace_maskdata[i].inlabel = is_rug_points[i];

    }

}




    


