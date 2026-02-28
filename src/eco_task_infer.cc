
/******************************************************************************
模块名　　　　： eco_task_infer
文件名　　　　： eco_task_infer.cpp
相关文件　　　： eco_task_infer.cpp
文件实现功能　： 业务任务类函数定义
作者　　　　　： 周峰
版本　　　　　：　1.0
-------------------------------------------------------------------------------
修改记录:
日  期        版本        修改人        创建人    修改内容
2022/01/08    1.0                     周峰         创建
2022/04/25    1.1        周峰　　　　　　　　　　　增加检测类以及双图片输入 3+2 模型
2022/06/02    1.2        周峰                  修改功能类，将原始分离的类指针合并成一个类指针，
                                              并在open中通过参数申请内
2022/09/09    1.3        周峰                  修改功能类，添加多模态输入
2023/01/09    1.4        周峰                  添加通过内参公式法测距 方法
******************************************************************************/


#include <unistd.h>
#include "eco_task_infer.h"
#include "postprocess.h"
#include "utils.h"
#include <fstream>
#include "eco_distance_Interface.h"
#include <fstream>

static int save_image = 1;
static std::vector<int64_t> rugsegtimes;  // 记录检测到地毯的时间戳 最大10帧
int64_t  rug_seg_time(-1);
static int cm_distance(-1);
static int cm_distance_ir(-1);
static int area_texture(-1);
static bool save_rgb_next = true; // 标志位：true表示下次存RGB，false表示下次存红外


namespace sweeper_ai
{

// 目标检测模型编号
const std::vector<int> vdetectconfigparams      = {0,  1,  2,  4,  5,  6,  7,  9, 11, 12, 13, 14, 15, 21};
// 语义分割，点检测模型编号
const std::vector<int> vsegconfigparams         = {3, 10};
// 分类识别模型编号
const std::vector<int> vclsconfigparams         = { };

const std::vector<int> virdetectconfigparams    = {21};

/// 这里初始化时不同模型的数量也得填写
EcoTaskInference::EcoTaskInference():
nirdetectmodelnum(1), ecoircamtargetdetect_(NULL),
ndetectmodelnum(14),   ecocamtargetdetect_(NULL),
nsegmodelnum(2),      ecocamtargetseg_(NULL), 
nclsmodelnum(0),      ecocamtargetcls_(NULL), 
nimgnum(16),          ecocamoutputresult_(NULL),
bflag(EM_RGB_AI_TYPE)
{

}

EcoTaskInference::~EcoTaskInference()
{


}

EcoEStatus EcoTaskInference::ecoTaskOpen(char * config_str)
{

    EcoEStatus ecoestatus(EStatus_Success);

    std::ifstream t(config_str);
    std::string   str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    rapidjson::Document document;
    document.Parse(config_str);
    // 解析配置文件
    if (document.HasParseError()) 
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "document.HasParseError() error in ecoTaskOpen" <<  ecoestatus << std::endl;
        ecoTaskClose();
        return ecoestatus;
    }

    if (document.HasMember("type"))
    {
       bflag =(EM_AI_TYPE)document["type"].GetInt(); 
       std::cout << "bflag = " << bflag << std::endl;
    }

if(bflag == EM_RGB_AI_TYPE)
{
/**** *目标检测* 申请对应个数的模型数组 *********************************************************************************************************************************************************/
    ndetectmodelnum = vdetectconfigparams.size();
    if (ndetectmodelnum > 0)
    {
        // 1 创建目标检测类
        ecocamtargetdetect_ = new EcoDetectInference[ndetectmodelnum];
        if (NULL == ecocamtargetdetect_)
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout  <<  "ecocamtargetdetect_ can't memory in ecoTaskOpen" <<  ecoestatus << std::endl;
            ecoTaskClose();
            return ecoestatus;
        }
    }

/*****  *语义分割* * 关键点检测* 申请对应个数的模型数组 **************************************************************************************************************************************************/
    nsegmodelnum = vsegconfigparams.size();
    if (nsegmodelnum > 0)
    {
        // 语义分割类
        ecocamtargetseg_ = new EcoSegInference[nsegmodelnum];
        if (NULL == ecocamtargetseg_)
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout  <<  "ecocamtargetseg_ can't memory in ecoTaskOpen" <<  ecoestatus << std::endl;
            ecoTaskClose();
            return ecoestatus;
        }
    }


/*******  *目标识别 (分类)*  申请对应个数的模型数组 ********************************************************************************************************************************************************/
    nclsmodelnum = vclsconfigparams.size();
    if (nclsmodelnum > 0)
    {
        //目标识别类
        ecocamtargetcls_ = new EcoObjectClsInference[nclsmodelnum];
        if (NULL == ecocamtargetcls_)
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout  <<  "ecocamtargetcls_ can't memory in ecoTaskOpen" <<  ecoestatus << std::endl;
            ecoTaskClose();
            return ecoestatus;
        }
    }


    rapidjson::Value modelsparams;
    if (document.HasMember("models"))
    {
       modelsparams = document["models"].GetArray(); 
    }
    // json文件中的 “models” 中有几个模型就是 modelsparams.Size()
    for (rapidjson::SizeType npa = 0; npa < modelsparams.Size(); npa++)
    {
        rapidjson::Value model_sparam = modelsparams[npa].GetObject();

        if (model_sparam.HasMember("id"))
        {
            // 根据json文件中设置的“id”的值来得到model_id
            int model_id = model_sparam["id"].GetInt();

/*******  * model_id  参数 属于目标检测 *  ********************************************************************************************************************************************************/
            if (find(vdetectconfigparams.begin(), vdetectconfigparams.end(), model_id) != vdetectconfigparams.end())
            {
                int id = find(vdetectconfigparams.begin(), vdetectconfigparams.end(), model_id) - vdetectconfigparams.begin();
      
                //　申请　目标检测类　的内存
                ecoestatus = ecocamtargetdetect_[id].ecoDetectOpen(model_sparam);
                if (ecoestatus)
                {
                    std::cout << "ecocamtargetdetect_[ "<< id << " ].ecoDetectOpen error in ecoTaskOpen" << ecoestatus << std::endl;
                    ecoTaskClose();
                    return ecoestatus;
                }

                std::cout << "ecocamtargetdetect_[ " << id << " ].ecoDetectOpen() sucess in ecoTaskOpen" << ecoestatus << std::endl;
                std::cout << std::endl << std::endl;
            }

/*******  * model_id  参数 属于分割与关键点检测 *  ********************************************************************************************************************************************************/

            else if (find(vsegconfigparams.begin(), vsegconfigparams.end(), model_id ) != vsegconfigparams.end())
            {
                int id = find(vsegconfigparams.begin(), vsegconfigparams.end(), model_id) - vsegconfigparams.begin();
           
                //　申请　目标检测类　的内存
                ecoestatus = ecocamtargetseg_[id].ecoSegOpen(model_sparam);
                if (ecoestatus)
                {
                    std::cout << "ecocamtargetseg_[ " << id << " ].ecoSegOpen() error in ecoTaskOpen" << ecoestatus << std::endl;
                    ecoTaskClose();
                    return ecoestatus;
                }

                std::cout << "ecocamtargetseg_[ " << id << " ].ecoSegOpen() sucess in ecoTaskOpen" << ecoestatus << std::endl;
                std::cout << std::endl << std::endl;
            
            }

/*******  * model_id  参数 属于分类识别 *  ********************************************************************************************************************************************************/

            else if (find( vclsconfigparams.begin(), vclsconfigparams.end(), model_id) != vclsconfigparams.end() )
            {

                int id = find( vclsconfigparams.begin( ), vclsconfigparams.end( ), model_id ) - vclsconfigparams.begin( );
           
                //　申请　目标识别类　的内存
                ecoestatus = ecocamtargetcls_[id].ecoObjectClsOpen(model_sparam);
                if (ecoestatus)
                {
                    std::cout << "ecocamtargetcls_[ " << id << ".ecoObjectClsOpen() error in ecoTaskOpen" << ecoestatus << std::endl;
                    ecoTaskClose();
                    return ecoestatus;
                }

                std::cout << "ecocamtargetcls_[ " << id << ".ecoObjectClsOpen() sucess in ecoTaskOpen" << ecoestatus << std::endl;
                std::cout << std::endl << std::endl;
            }

// /*******  * model_id  参数 无效 *  ********************************************************************************************************************************************************/

//             else
//             {
//                 ecoestatus = EStatus_InvalidParameter;
//                 std::cout << "model ID is Invalid in ecoTaskOpen" << ecoestatus << std::endl;
//                 ecoTaskClose();
//                 return ecoestatus;

//             }

        }
    }

/******** *结果输出* *****************************************************************************************************************************************************/

    //　申请模型结果内存
    nimgnum = ndetectmodelnum + nsegmodelnum + nclsmodelnum;

    ecocamoutputresult_ = new EcoInstanceObjectSeg[nimgnum];
    if (NULL == ecocamoutputresult_)
	{
		ecoestatus = EStatus_OutOfMemory;
		std::cout  <<  "ecocamoutputresult_ can't memory" <<  ecoestatus << std::endl;
        ecoTaskClose();
        return ecoestatus;
	}

    for (size_t i = 0; i < nimgnum; i++)
    {
        ecocamoutputresult_[i].mask = cv::Mat(192, 256, CV_8UC1, cv::Scalar(0));
    }

/**************************************************************   镜头内参数读取   *************************************************************************************************/
    ////读取相机内参
    if(document.HasMember("private"))
    {
        rapidjson::Value camparams = document["private"].GetObject();
        if (camparams.HasMember("inner_params"))
        {
            std::string RGBD1_cam_yaml_path = camparams["inner_params"].GetString();
            if(-1 == access(RGBD1_cam_yaml_path.c_str(), 0))
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "have no RGBD1_cam_yaml_path file " << RGBD1_cam_yaml_path << std::endl;
                ecoTaskClose();
                return ecoestatus;
            }

            // 读取相机内参
            ecoestatus = load_caminfo(0, RGBD1_cam_yaml_path, RGBDup);
            if(ecoestatus != EStatus_Success)
            {
                ecoTaskClose();
                return ecoestatus;
            }

            std::cout <<  "camid = RGBD1 " << "  fx = "  <<  RGBDup.RGB.fx
                << "  fy = "  << RGBDup.RGB.fy  << "  cx = "  <<  RGBDup.RGB.cx
                << "  cy = " <<RGBDup.RGB.cy    << std::endl; 

        }
        if (camparams.HasMember("IR_inner_params"))
        {
            std::string RGBD1_cam_yaml_path = camparams["IR_inner_params"].GetString();
            if(-1 == access(RGBD1_cam_yaml_path.c_str(), 0))
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "have no IR_inner_params file " << RGBD1_cam_yaml_path << std::endl;
                ecoTaskClose();
                return ecoestatus;
            }

            // 读取相机内参
            ecoestatus = load_caminfo(1, RGBD1_cam_yaml_path, RGBDup);
            if(ecoestatus != EStatus_Success)
            {
                ecoTaskClose();
                return ecoestatus;
            }

            std::cout <<  "camid = IR " << "  fx = "  <<  RGBDup.IR.fx
                << "  fy = "  << RGBDup.IR.fy  << "  cx = "  <<  RGBDup.IR.cx
                << "  cy = " << RGBDup.IR.cy   << "  angle = " << RGBDup.ir_rangle  << "  height = " << RGBDup.ir_Height << std::endl; 

        }
    }


/**************************************************************   测距文件生成   *************************************************************************************************/
    rapidjson::Value distance_obj;
    // if (document.HasMember("distance"))
    {
        // distance_obj = document["distance"].GetObject();
        // if (distance_obj.HasMember("distance_table"))
        {
            std::string distance_name = "/data/zhoufeng/2435/distance_table";

            if(-1 == access(distance_name.c_str(), 0))
            {
                distance_name = "/data/distance_table";
            }
            std::cout << "distance_name = " << distance_name << std::endl;
            
            if(-1 == access(distance_name.c_str(), 0))
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "have no distance_table file /data/distance_table " << distance_name << std::endl;
                ecoTaskClose();
                return ecoestatus;
            }
            int Flag=0;
            uint8_t* table_data = NULL;
            int maxRowInfo[4]={0,0};
            const int IMG_ROW_RANGE[]={450,940};
            const int IMG_COL_RANGE[]={30,1250};
            const int table_width = IMG_COL_RANGE[1] - IMG_COL_RANGE[0] + 1;
            int table_bufLen=2*(IMG_ROW_RANGE[1]-IMG_ROW_RANGE[0]+1)*(IMG_COL_RANGE[1]-IMG_COL_RANGE[0]+1);
            int table_size = distance_name.length() + 1;
            char table[table_size];
            strcpy(table, distance_name.c_str()); 
            FILE *file = fopen(table , "rb");
            if(file != NULL)
            {
                table_data = (uint8_t *) malloc(table_bufLen);
                if (table_data == NULL) {
                    printf("cannot malloc table_data, table_data is NULL\n");
                    fclose(file);
                    ecoestatus = EStatus_InvalidParameter;
                    ecoTaskClose();
                    return ecoestatus;
                }
                fseek(file, 0, SEEK_END); // seek to end of file 设置文件指针指向文件结尾
                int fileSize = ftell(file); // get current file pointer 获取文件结尾到文件头的长度 获取文件容量
                if (fileSize == -1) {
                    fclose(file);
                    std::cout << "distance_table file size is fault"<< std::endl;
                    ecoestatus = EStatus_InvalidParameter;
                    ecoTaskClose();
                    return ecoestatus;
                }
                rewind(file); // 将文件的指针指向文件开头
                printf("table_bufLen:%d %d\n" , table_bufLen , fileSize);
                int readed = fread(table_data , 1 , table_bufLen , file);
                if(readed != table_bufLen)
                {
                    // delete table_data;
                    free(table_data);
                    table_data = NULL;
                    fclose(file);
                    printf("file size not match\n");
                    ecoestatus = EStatus_InvalidParameter;
                    ecoTaskClose();
                    return ecoestatus;
                }

                for (int i = IMG_ROW_RANGE[0]; i <=IMG_ROW_RANGE[1]; ++i) {
                    for (int j = IMG_COL_RANGE[0]; j <=IMG_COL_RANGE[1] ; ++j) {//2 * ((pixel_row - IMG_ROW_RANGE[0]) * table_width + ((pixel_col - i) - IMG_COL_RANGE[0]))
                        if (table_data[2 * ((i - IMG_ROW_RANGE[0]) * table_width + (j- IMG_COL_RANGE[0]))]!=0) {
                            maxRowInfo[0]=i; maxRowInfo[1]=j;
                            goto part1;
                        }
            
                    }
                }
                part1:
                ;
            
                for (int i = IMG_ROW_RANGE[1]; i >=IMG_ROW_RANGE[0]; i--) {
                    for (int j = IMG_COL_RANGE[1]; j >=IMG_COL_RANGE[0] ; j--) {//2 * ((pixel_row - IMG_ROW_RANGE[0]) * table_width + ((pixel_col - i) - IMG_COL_RANGE[0]))
                        if (table_data[2 * ((i - IMG_ROW_RANGE[0]) * table_width + (j- IMG_COL_RANGE[0]))]!=0) {
                            maxRowInfo[2]=i; maxRowInfo[3]=j;
                            goto part2;
                        }
            
                    }
                }
                part2:
                ;
                fclose(file);
                
                ecoDistanceInit(table_data,maxRowInfo,RGBDup.RGB.fx, RGBDup.RGB.fy, RGBDup.RGB.cx, RGBDup.RGB.cy, Flag, cm_distance);
                std::cout<<"222 "<< " cm_distance = " << cm_distance << std::endl;

                // table_data内存释放
                if(table_data != NULL)
                {
                    free(table_data);
                    table_data = NULL;
                }

            }else{
                printf("cannot open file:%s\n" , table);
                ecoestatus = EStatus_InvalidParameter;
                ecoTaskClose();
                return ecoestatus;
                
            }
        }
    }

    std::string distance_name = "/data/sl_distance_table";
    std::cout << "distance_name = " << distance_name << std::endl;
    
    if(-1 == access(distance_name.c_str(), 0))
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout << "have no distance_table file /data/sl_distance_table " << distance_name << std::endl;
        ecoTaskClose();
        return ecoestatus;
    }
    else
    {
        int Flag=0;
        uint8_t* table_data = NULL;
        const int IMG_ROW_RANGE[]={360,720};
        const int IMG_COL_RANGE[]={30,1250};
        const int table_width = IMG_COL_RANGE[1] - IMG_COL_RANGE[0] + 1;
        int table_bufLen=2*(IMG_ROW_RANGE[1]-IMG_ROW_RANGE[0]+1)*(IMG_COL_RANGE[1]-IMG_COL_RANGE[0]+1);
        int maxRowInfo[1280*2];
        int table_size = distance_name.length() + 1;
        char table[table_size];
        strcpy(table, distance_name.c_str()); 
        FILE *file = fopen(table , "rb");
        if(file != NULL)
        {
            table_data = (uint8_t *) malloc(table_bufLen);
            if (table_data == NULL) {
                printf("cannot malloc table_data, table_data is NULL\n");
                fclose(file);
                ecoestatus = EStatus_InvalidParameter;
                ecoTaskClose();
                return ecoestatus;
            }
            fseek(file, 0, SEEK_END); // seek to end of file 设置文件指针指向文件结尾
            int fileSize = ftell(file); // get current file pointer 获取文件结尾到文件头的长度 获取文件容量
            if (fileSize == -1) {
                fclose(file);
                std::cout << "distance_table file size is fault"<< std::endl;
                ecoestatus = EStatus_InvalidParameter;
                ecoTaskClose();
                return ecoestatus;
            }
            rewind(file); // 将文件的指针指向文件开头
            printf("table_bufLen:%d %d\n" , table_bufLen , fileSize);
            int readed = fread(table_data , 1 , table_bufLen , file);
            if(readed != table_bufLen)
            {
                // delete table_data;
                free(table_data);
                table_data = NULL;
                fclose(file);
                printf("file size not match\n");
                ecoestatus = EStatus_InvalidParameter;
                ecoTaskClose();
                return ecoestatus;
            }

            for (int col = 0; col < 1280; ++col) {
                maxRowInfo[2 * col] = -1;
                maxRowInfo[2 * col + 1] = -1;

                if (col >= IMG_COL_RANGE[0] && col <= IMG_COL_RANGE[1]){
                    for (int row = IMG_ROW_RANGE[0]; row <= IMG_ROW_RANGE[1]; ++row) {
                        int index = 2 * ((row - IMG_ROW_RANGE[0]) * table_width + (col- IMG_COL_RANGE[0]));
                        if (table_data[index] != 0) {
                            maxRowInfo[2 * col] = row;
                            break;
                        }
                    }

                    for (int row = IMG_ROW_RANGE[1]; row >= IMG_ROW_RANGE[0]; --row) {
                        int index = 2 * ((row - IMG_ROW_RANGE[0]) * table_width + (col- IMG_COL_RANGE[0]));
                        if (table_data[index] != 0) {
                            maxRowInfo[2 * col + 1] = row;
                            break;
                        }
                    }
                }
            }

            fclose(file);
            
            ecoDistanceInit_ir(table_data,maxRowInfo,RGBDup.IR.fx, RGBDup.IR.fy, RGBDup.IR.cx, RGBDup.IR.cy, Flag, cm_distance_ir);
            std::cout<<"222 "<< " cm_distance_ir = " << cm_distance_ir << std::endl;

            // table_data内存释放
            if(table_data != NULL)
            {
                free(table_data);
                table_data = NULL;
            }

        }else{
            printf("cannot open file:%s\n" , table);
            ecoestatus = EStatus_InvalidParameter;
            ecoTaskClose();
            return ecoestatus;
        }
    }

}
else if(bflag == EM_IR_TYPE)     //   红外图 AI 功能
{

/**** *目标检测* 申请对应个数的模型数组 *********************************************************************************************************************************************************/
    nirdetectmodelnum = virdetectconfigparams.size();
    if (nirdetectmodelnum > 0)
    {
        // 1 创建目标检测类
        ecoircamtargetdetect_ = new EcoDetectInference[nirdetectmodelnum];
        if (NULL == ecoircamtargetdetect_)
        {
            ecoestatus = EStatus_OutOfMemory;
            std::cout  <<  "ecoircamtargetdetect_ can't memory in ecoTaskOpen" <<  ecoestatus << std::endl;
            ecoTaskClose();
            return ecoestatus;
        }
    }

    rapidjson::Value modelsparams;
    if (document.HasMember("models"))
    {
       modelsparams = document["models"].GetArray(); 
    }

    // json文件中的 “models” 中有几个模型就是 modelsparams.Size()
    for (rapidjson::SizeType npa = 0; npa < modelsparams.Size(); npa++)
    {
        rapidjson::Value model_sparam = modelsparams[npa].GetObject();

        if (model_sparam.HasMember("id"))
        {
            // 根据json文件中设置的“id”的值来得到model_id
            int model_id = model_sparam["id"].GetInt();
            if(model_id == 21)
            {
                ecoestatus = ecoircamtargetdetect_[0].ecoDetectOpen(model_sparam);
                if (ecoestatus)
                {
                    std::cout << "ecoircamtargetdetect_[ "<< 0 << " ].ecoDetectOpen error in ecoTaskOpen" << ecoestatus << std::endl;
                    ecoTaskClose();
                    return ecoestatus;
                }

                std::cout << "ecoircamtargetdetect_[ " << 0 << ".ecoObjectClsOpen() sucess in ecoTaskOpen" << ecoestatus << std::endl;
                std::cout << std::endl << std::endl;
            }
        }
    }

    //　申请模型结果内存
    nimgnum = nirdetectmodelnum;

    ecocamoutputresult_ = new EcoInstanceObjectSeg[nimgnum];
    if (NULL == ecocamoutputresult_)
	{
		ecoestatus = EStatus_OutOfMemory;
		std::cout  <<  "ecocamoutputresult_ can't memory" <<  ecoestatus << std::endl;
        ecoTaskClose();
        return ecoestatus;
	}

/**************************************************************   镜头内参数读取   *************************************************************************************************/
    ////读取红外相机内参
    if(document.HasMember("private"))
    {
        rapidjson::Value camparams = document["private"].GetObject();
        if (camparams.HasMember("IR_inner_params"))
        {
            std::string RGBD1_cam_yaml_path = camparams["IR_inner_params"].GetString();
            if(-1 == access(RGBD1_cam_yaml_path.c_str(), 0))
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "have no IR_inner_params file " << RGBD1_cam_yaml_path << std::endl;
                ecoTaskClose();
                return ecoestatus;
            }

            // 读取相机内参
            ecoestatus = load_caminfo(1, RGBD1_cam_yaml_path, RGBDup);
            if(ecoestatus != EStatus_Success)
            {
                ecoTaskClose();
                return ecoestatus;
            }

            std::cout <<  "camid = RGBD1 " << "  fx = "  <<  RGBDup.RGB.fx
                << "  fy = "  << RGBDup.RGB.fy  << "  cx = "  <<  RGBDup.RGB.cx
                << "  cy = " << RGBDup.RGB.cy   << "  angle = " << RGBDup.rangle  << "  height = " << RGBDup.Height << std::endl; 

        }
    }
}


    rug_mask = cv::Mat(25+7, 80, CV_8UC1, cv::Scalar(255));

    return EStatus_Success;
}



EcoEStatus EcoTaskInference::ecoTaskInfer(const ImageDatas &input_data)
{
    EcoEStatus ecoestatus(EStatus_Success);
    int model_index(-1);
    model_index = input_data.input_image[0].model_index;

    static std::deque<std::unordered_set<int>> coords_w_queue;  // yp_0211 地毯多帧匹配队列
    static std::deque<int> time_queue;

    if (NULL == input_data.input_image )
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "NULL != input_data.input_image is error in EcoTaskInference::ecoTaskInfer" <<  ecoestatus <<std::endl;
        return ecoestatus;
    }

    // 输出结果转换
    EcoInstanceObjectSeg* ecoCamOutputResult = &ecocamoutputresult_[model_index];  //　输出结果
    if (NULL == ecoCamOutputResult)
    {
        ecoestatus = EStatus_InvalidParameter;
        std::cout  <<  "NULL == ecoCamOutputResult[ " << model_index << " ] in ecoTaskInfer" <<  ecoestatus <<std::endl;
        return ecoestatus;
    }

    for (int imgid = 0; imgid < input_data.num_img; imgid++)
    {

        ecoCamOutputResult->frame_id             = input_data.input_image[imgid].frame_id;
        ecoCamOutputResult->model_id             = input_data.input_image[imgid].model_id;
        ecoCamOutputResult->st                   = input_data.input_image[imgid].st;
        ecoCamOutputResult->maskdata.clear();
        ecoCamOutputResult->ecogroundobjects     = NULL;
        ecoCamOutputResult->imagecls.label       = EM_BACKGROUND;
        ecoCamOutputResult->imagecls.fconfidence = -1;

        //////modelSwitch为0时是三个模型均未开，为1时是
        int modelSwitch = static_cast<int>(input_data.input_image[imgid].switchON) & 255;

/*******************************************************************   去畸变图像 512*384   非去畸变 1280*960 ********************************************************************************************/
        // 镜头编号为　０　的图像数据--对应的输入的图片是未曾去畸变的
        ////镜头编号为　1　的图像数据--对应的输入的图片是去畸变的
        if (input_data.input_image[imgid].frame_id == 1)
        {

/**********************************************************     障碍物（多任务）识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 0)
            {
                std::vector<cv::Mat>            input_images;   
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                //　地面障碍物识别
                EcoDetectInference* ecoObsRecogdetect_ = &ecocamtargetdetect_[0];  
                if (!ecoObsRecogdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoObsRecogdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }  

                // inference
                ecoestatus = ecoObsRecogdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoObsRecogdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }
                
                // 上镜头静态目标检测结果输出
                ecoCamOutputResult->ecogroundobjects = ecoObsRecogdetect_->getDetectObjects();

                // 障碍物测距结果
                topcamstataicdectect(0, ecoCamOutputResult, RGBDup, robot_lds_data, imu);


                // 判断是否存在级联模型--这里获取到after_model_ids中有值的话，从std::vector<int> after_model_ids中获取到其级联了哪些模型
                ////这个模型没有级联模型，这块不用写
                if(ecoObsRecogdetect_->getaftermodelids().size() > 0)
                {

                }

                std::string binary_dir = "/data/autostart/0/";
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_seg.jpg";
                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                if(-1 != access("/data/autostart/image/0/", 0))
                {
                    ecoObsRecogdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/0/", ecoCamOutputResult->mask);
                }
                

                input_images.clear();

            }

/**********************************************************     家具识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 1)
            {
                std::vector<cv::Mat>            input_images;   
                std::vector<std::vector<float>> robot_lds_data(10, std::vector<float>(0, 0));
                std::vector<int>                imu;
                input_images.clear();
                for(int nvec = 0; nvec < 10; nvec++)
                {
                    robot_lds_data[nvec].clear();
                }
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/1/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_1_.jpg";
                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                //　家具识别目标检测算法
                EcoDetectInference* ecoFurnRecogdetect_ = &ecocamtargetdetect_[1];  
                if (!ecoFurnRecogdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoFurnRecogdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }  

                // inference
                ecoestatus = ecoFurnRecogdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoFurnRecogdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 上镜头静态目标检测结果输出
                ecoCamOutputResult->ecogroundobjects = ecoFurnRecogdetect_->getDetectObjects();

                // LDS 点云转图像坐标系
                std::vector<float> Tcl;
                lds2pixel(robot_lds_data, input_data.input_image[imgid].st.timestamp, 
                input_data.input_image[imgid].st, input_data.input_image[imgid].ldsPointsData, input_data.input_image[imgid].slPointsData,
                RGBDup.RGB, Tcl);
                // 测距结果
                topcamstataicdectect(1, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                
                if(-1 != access("/data/autostart/image/1/", 0))
                {
                    ////临时写个保存结果图片为了测试
                    ecoFurnRecogdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/1/", ecoCamOutputResult->mask);
                }  

                input_images.clear();

            }

/**********************************************************    地毯分割  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 3)
            {
                std::vector<cv::Mat>            input_images;  
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                if(!rug_mask.empty())
                {
                    rug_mask.release();
                    rug_mask = cv::Mat(25+7, 80, CV_8UC1, cv::Scalar(255));
                }

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "RGB_image_src.empty() in ecodownlinetargetseg_" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/3/";
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_img.jpg";
                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src); 

                // 输入图片 ROI
                cv::Rect down_seg_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                // freespace 语义分割
                EcoSegInference* ecoFreespacetargetseg_ = &ecocamtargetseg_[0];    
                if (!ecoFreespacetargetseg_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoFreespacetargetseg_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }

                // inference
                ecoestatus = ecoFreespacetargetseg_->ecoSegInfer(input_images, down_seg_roi, cm_distance);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoFreespacetargetseg_->ecoSegInfer(input_images, down_seg_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 内部标签转对外标签，添加深度信息
                topcamstataicseg(3, ecoFreespacetargetseg_->getSegMasks(), RGBDup, robot_lds_data, imu, rug_mask);

                //添加结构光数据进行AI地毯辅助判断
                bool carpet_mask_valid_flag = false;
                std::vector<std::vector<float>> linelaser_to_cam;
                carpetFreespacePointToLinesaserFusion(ecoFreespacetargetseg_->getSegMasks()->mask, input_data.input_image[imgid].slPointsData, RGBDup.RGB,carpet_mask_valid_flag,
                    linelaser_to_cam,ecoFreespacetargetseg_->linelaser_ground_points_colmean_,ecoFreespacetargetseg_->linelaser_z_grayscale_average);
                secondVerificationCheckCarpetMaskVdlidFalg(ecoFreespacetargetseg_->carpet_linelaser_map_probability, ecoFreespacetargetseg_->carpet_linelaser_valid_pose_vec, input_data.input_image[imgid].slPointsData.status.x,input_data.input_image[imgid].slPointsData.status.y,input_data.input_image[imgid].slPointsData.status.Qz,ecoFreespacetargetseg_->getSegMasks(),
                    carpet_mask_valid_flag);

                std::string binary_dir1 = "/data/autostart/003/";
                if (access(binary_dir1.c_str(), 0) == 0) 
                {
                    std::vector<float> single_raw_linelsaer_data;
                    single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.status.x);
					single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.status.y);
					single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.status.Qz);
					// single_raw_linelsaer_data.push_back((int)input_data.input_image[imgid].slPointsData.idx);
                    single_raw_linelsaer_data.push_back((int)carpet_mask_valid_flag);
                    for(int i=0; i< SL_POINT_CNT; i++)
					{
						single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.slPoint[i].x);
						single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.slPoint[i].y);
						single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.slPoint[i].z);
						single_raw_linelsaer_data.push_back(input_data.input_image[imgid].slPointsData.slPoint[i].grayscales);
					}
                    std::string save_path = binary_dir1 + std::to_string(input_data.input_image[imgid].st.timestamp) + ".bin";
					std::ofstream binaryFile(save_path, std::ios::binary);
					if (binaryFile.is_open()) {
						binaryFile.write((char*)&single_raw_linelsaer_data[0], single_raw_linelsaer_data.size() * sizeof(float));
						binaryFile.close();
					}

                    // std::vector<float> single_raw_linelsaer_data_;
                    // single_raw_linelsaer_data_.push_back(ecoFreespacetargetseg_->linelaser_ground_points_colmean_.size());
                    // for(int ii = 0; ii < ecoFreespacetargetseg_->linelaser_ground_points_colmean_.size(); ii++)
                    // {
                    //     single_raw_linelsaer_data_.push_back(ecoFreespacetargetseg_->linelaser_ground_points_colmean_[ii]);
                    // }
                    // single_raw_linelsaer_data_.push_back(ecoFreespacetargetseg_->linelaser_z_grayscale_average.size());
                    // for(int ii = 0; ii < ecoFreespacetargetseg_->linelaser_z_grayscale_average.size(); ii++)
                    // {
                    //     single_raw_linelsaer_data_.push_back(ecoFreespacetargetseg_->linelaser_z_grayscale_average[ii][0]);
                    //     single_raw_linelsaer_data_.push_back(ecoFreespacetargetseg_->linelaser_z_grayscale_average[ii][1]);
                    // }
                    // std::string save_path2 = binary_dir1 + std::to_string(input_data.input_image[imgid].st.timestamp) + ".bata";
					// std::ofstream binaryFile2(save_path2, std::ios::binary);
					// if (binaryFile2.is_open()) {
					// 	binaryFile2.write((char*)&single_raw_linelsaer_data_[0], single_raw_linelsaer_data_.size() * sizeof(float));
					// 	binaryFile2.close();
					// }

                    // std::string binary_path_SEG1 = binary_dir1 + std::to_string(input_data.input_image[imgid].st.timestamp) + "_.jpg";
                    // cv::Mat carpet_mask = ecoFreespacetargetseg_->getSegMasks()->mask.clone();
                    // cv::imwrite(binary_path_SEG1, carpet_mask);

                    cv::Mat raw_img = RGB_image_src.clone();
                    std::string binary_path_SEG = binary_dir1 + std::to_string(input_data.input_image[imgid].st.timestamp) + ".jpg";
                    cv::imwrite(binary_path_SEG, raw_img);

                    //保存mask的测距信息，离线叠图
                    // std::vector<float> single_maskdata_distance;
                    // for(int kk = 0; kk < ecoFreespacetargetseg_->getSegMasks()->maskdata.size(); kk++)
                    // {
                    //     single_maskdata_distance.push_back(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].mappos.x);
                    //     single_maskdata_distance.push_back(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].mappos.y);
                    //     single_maskdata_distance.push_back(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.x);
                    //     single_maskdata_distance.push_back(ecoFreespacetargetseg_->getSegMasks()->maskdata[kk].keypoint.y);
                    // }
                    // std::string save_path1 = binary_dir1 + std::to_string(input_data.input_image[imgid].st.timestamp) + ".data";
					// std::ofstream binaryFile1(save_path1, std::ios::binary);
					// if (binaryFile1.is_open()) {
					// 	binaryFile1.write((char*)&single_maskdata_distance[0], single_maskdata_distance.size() * sizeof(float));
					// 	binaryFile1.close();
                    // }
                }
                // if(!carpet_mask_valid_flag)
                // {
                //     ecoFreespacetargetseg_->getSegMasks()->maskdata.clear();
                // }
                // std::cout << "carpet_mask_valid_flag: " << carpet_mask_valid_flag << std::endl;

                // 分割结果输出
                ecoCamOutputResult->mask     = ecoFreespacetargetseg_->getSegMasks()->mask;  
                ecoCamOutputResult->maskdata     = ecoFreespacetargetseg_->getSegMasks()->maskdata;  

                // 保存图像结果 
                if(-1 != access("/data/autostart/image/3/", 0))
                {   
                    std::string  binary_dir = "/data/autostart/image/3/mask-";
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_mask.jpg";
                    cv::imwrite(binary_path_SEG, ecoCamOutputResult->mask);

                    binary_dir = "/data/autostart/image/3/bev-";
                    binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_bev.jpg";
                    cv::imwrite(binary_path_SEG, rug_mask);

                    ecoFreespacetargetseg_->showSegMasks(ecoCamOutputResult->maskdata, RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/3/3-");
                }


                // 保存图像结果 
                if(-1 != access("/data/autostart/image/03/", 0))
                {   
                    if(abs(rug_seg_time - input_data.input_image[imgid].st.timestamp) > 2000)
                    {
                        rug_seg_time = input_data.input_image[imgid].st.timestamp;
                        std::string  binary_dir = "/data/autostart/image/03/3--";
                        std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                        "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_mask.jpg";
                        cv::imwrite(binary_path_SEG, ecoCamOutputResult->mask);

                        ecoFreespacetargetseg_->showSegMasks(ecoCamOutputResult->maskdata, RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/03/3-");
                    }
                }

                input_images.clear();
            }

/**********************************************************    通用接地线  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 10)
            {
                std::vector<cv::Mat>            input_images;  
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "RGB_image_src.empty() in ecodownlinetargetseg_" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/10/";
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_img.jpg";
                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src); 

                // 输入图片 ROI
                cv::Rect down_seg_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);    
                // groundpoints 语义分割
                EcoSegInference* ecoGroundPointsseg_ = &ecocamtargetseg_[1];
                if (!ecoGroundPointsseg_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoGroundPointsseg_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }

                // inference
                ecoestatus = ecoGroundPointsseg_->ecoSegInfer(input_images, down_seg_roi, cm_distance);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoGroundPointsseg_->ecoSegInfer(input_images, down_seg_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 内部标签转对外标签，添加深度信息
                topcamstataicseg(10, ecoGroundPointsseg_->getSegMasks(), RGBDup, robot_lds_data, imu, rug_mask);

                ecoCamOutputResult->maskdata     = ecoGroundPointsseg_->getSegMasks()->maskdata;  

                // groundpoint2rug(ecoCamOutputResult, ecoGroundPointsseg_, ecoFreespacetargetseg_, rug_mask);

                // 保存图像结果 
                if(-1 != access("/data/autostart/image/10/", 0))
                {   
                    ecoGroundPointsseg_->showSegMasks(ecoCamOutputResult->maskdata, RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/10/10-");
                }

                input_images.clear();
            }

/**********************************************************     低矮区域识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 4)
            {
                std::vector<cv::Mat>            input_images;   
                std::vector< std::vector<float> > robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }
                input_images.push_back(RGB_image_src);  

                std::string binary_dir = "/data/autostart/4/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    if (save_rgb_next) 
                    {
                        cv::Mat RGB_image_src_clone = RGB_image_src.clone();
                        std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                        "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_" + std::to_string(area_texture) + "_4_.jpg";

                        cv::imwrite(binary_path_SEG, RGB_image_src_clone);
                        save_rgb_next = false;
                    }
                    else
                    {
                        if(input_data.input_image[imgid].image_ir_data_addr != NULL)
                        {
                            cv::Mat ir_image_src(input_data.input_image[imgid].image_depth_height, input_data.input_image[imgid].image_depth_width, CV_8UC3, (void *)input_data.input_image[imgid].image_ir_data_addr);
                            cv::Mat ir_image_src_clone = cv::Mat(384, 640, CV_8UC3, cv::Scalar(114, 114, 114));;
                            eco_resize(ir_image_src, ir_image_src_clone, 640, 384, EM_RGA_NORMAL_RESIZE);
                            std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                            "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_" + std::to_string(area_texture) + "_21_.jpg";

                            cv::Mat single_channel;
                            cv::extractChannel(ir_image_src_clone, single_channel, 0);

                            cv::imwrite(binary_path_SEG, single_channel);
                        }
                        else
                        {
                            std::cout << "input_data.input_image[imgid].image_ir_data_addr is NULL" << std::endl;
                            cv::Mat RGB_image_src_clone = RGB_image_src.clone();
                            std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                            "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_" + std::to_string(area_texture) + "_4_.jpg";

                            cv::imwrite(binary_path_SEG, RGB_image_src_clone);
                        }
                        save_rgb_next = true;
                    }
                }
                continue;

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                EcoDetectInference* ecoLowRiseAreasDetect_ = &ecocamtargetdetect_[3]; 
                if (!ecoLowRiseAreasDetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoLowRiseAreasDetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                } 

                // inference
                ecoestatus = ecoLowRiseAreasDetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoLowRiseAreasDetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 上镜头静态目标检测结果输出
                ecoCamOutputResult->ecogroundobjects = ecoLowRiseAreasDetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataiccls(4, ecoCamOutputResult, RGBDup, robot_lds_data, imu);

                if(-1 != access("/data/autostart/image/4/", 0))
                {
                    ////临时写个保存结果图片为了测试
                    ecoLowRiseAreasDetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/4/", ecoCamOutputResult->mask);
                }  

                input_images.clear();
            }

/**********************************************************    地面材质分类  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 5)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }
                std::string binary_dir = "/data/autostart/5/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_5_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                // cv::Rect top_rect_roi(RGB_image_src.cols * 0.3, RGB_image_src.rows * 0.5, RGB_image_src.cols * 0.5, RGB_image_src.rows * 0.3);
                cv::Rect top_rect_roi(RGB_image_src.cols * 0.3, RGB_image_src.rows * 0.6, RGB_image_src.cols * 0.5, RGB_image_src.rows * 0.4);
                EcoDetectInference* ecoGroundMaterDetect_ = &ecocamtargetdetect_[4];  
                if (!ecoGroundMaterDetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoGroundMaterDetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }       
                // inference
                ecoestatus = ecoGroundMaterDetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoGroundMaterDetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 上镜头静态目标检测结果输出
                ecoCamOutputResult->ecogroundobjects = ecoGroundMaterDetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataiccls(5, ecoCamOutputResult, RGBDup, robot_lds_data, imu);

                if(-1 != access("/data/autostart/image/5/", 0))
                {
                    ////临时写个保存结果图片为了测试
                    ecoGroundMaterDetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/5/", ecoCamOutputResult->mask);
                }  

                input_images.clear();
            }

/**********************************************************     颗粒物 + 污渍 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 6)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();


                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/6/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_6_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 208, RGB_image_src.cols, RGB_image_src.rows - 208);

                EcoDetectInference* ecocamPMdetect_ = &ecocamtargetdetect_[5];  //　颗粒物识别
                if (!ecocamPMdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamPMdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                std::cout << "$$$$$$$$:modelSwitch: " << modelSwitch << std::endl;
                // inference  颗粒物开关二进制表示
                ecoestatus = ecocamPMdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamPMdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }
                // 颗粒物识别结果输出
                ecoCamOutputResult->ecogroundobjects = ecocamPMdetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataicdectect(6, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/6/", 0))
                {
                    ecocamPMdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/6/", ecoCamOutputResult->mask);
                }
                if (ecoCamOutputResult->ecogroundobjects->ngroundobjectnum > 0){
                    ecoCamOutputResult->mask = RGB_image_src;
                }
                input_images.clear();
            }

            /**********************************************************     宠物 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 11)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }
                input_images.push_back(RGB_image_src);  
                // cv::imwrite("./image.jpg",RGB_image_src);

                std::string binary_dir = "/data/autostart/11/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_11_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                EcoDetectInference* ecoAnimalcamObjdetect_ = &ecocamtargetdetect_[8];  //　颗粒物识别
                if (!ecoAnimalcamObjdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoAnimalcamObjdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                ecoestatus = ecoAnimalcamObjdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoAnimalcamObjdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 颗粒物识别结果输出
                ecoCamOutputResult->ecogroundobjects = ecoAnimalcamObjdetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataicdectect(11, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/11/", 0))
                {
                    ecoAnimalcamObjdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/11/", ecoCamOutputResult->mask);
                }
                if (ecoCamOutputResult->ecogroundobjects->ngroundobjectnum > 0){
                    ecoCamOutputResult->mask = RGB_image_src;
                }
                input_images.clear();

            }

            /**********************************************************   障碍物 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 12)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }
                input_images.push_back(RGB_image_src);  
                // cv::imwrite("./image.jpg",RGB_image_src);

                std::string binary_dir = "/data/autostart/12/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_12_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                EcoDetectInference* ecoObstaclecamObjdetect_ = &ecocamtargetdetect_[9];
                if (!ecoObstaclecamObjdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoObstaclecamObjdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                ecoestatus = ecoObstaclecamObjdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoObstaclecamObjdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 颗粒物识别结果输出
                ecoCamOutputResult->ecogroundobjects = ecoObstaclecamObjdetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataicdectect(12, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/12/", 0))
                {
                    ecoObstaclecamObjdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/12/", ecoCamOutputResult->mask);
                }
                input_images.clear();
            }

            /**********************************************************     毛絮 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 13)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/13/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_13_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 208, RGB_image_src.cols, RGB_image_src.rows - 208);

                EcoDetectInference* ecocamLintdetect_ = &ecocamtargetdetect_[10];
                if (!ecocamLintdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamLintdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                ecoestatus = ecocamLintdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamLintdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }
                // 识别结果输出
                ecoCamOutputResult->ecogroundobjects = ecocamLintdetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataicdectect(13, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/13/", 0))
                {
                    ecocamLintdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/13/", ecoCamOutputResult->mask);
                }
                input_images.clear();
            }

            /**********************************************************     干涸污渍 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 14)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/14/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_14_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 208, RGB_image_src.cols, RGB_image_src.rows - 208);

                EcoDetectInference* ecocamDryStaindetect_ = &ecocamtargetdetect_[11];
                if (!ecocamDryStaindetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamDryStaindetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                
                if (input_data.input_image[imgid].spotAreas_lock != NULL)
                {
                    printf("[debug]111库程序: 锁地址 = %p\n", input_data.input_image[imgid].spotAreas_lock);
                    printf("[debug]111指针 = %p\n", input_data.input_image[imgid].spotAreas);
                    pthread_mutex_lock(input_data.input_image[imgid].spotAreas_lock);
                    area_texture = checkPoseInPureTextureArea(*input_data.input_image[imgid].spotAreas, input_data.input_image[imgid].st);
                    pthread_mutex_unlock(input_data.input_image[imgid].spotAreas_lock);
                    printf("[debug]222库程序: 锁地址 = %p\n", input_data.input_image[imgid].spotAreas_lock);
                }
                else
                {
                    area_texture = 0;
                }

                ecoestatus = ecocamDryStaindetect_->ecoDetectInfer(input_images, top_rect_roi, area_texture);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamDryStaindetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }
                // 识别结果输出
                ecoCamOutputResult->ecogroundobjects = ecocamDryStaindetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataicdectect(14, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/14/", 0))
                {
                    ecocamDryStaindetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/14/", ecoCamOutputResult->mask);
                }
                if (ecoCamOutputResult->ecogroundobjects->ngroundobjectnum > 0){
                    ecoCamOutputResult->mask = RGB_image_src;
                }
                input_images.clear();
            }

            /**********************************************************  纯色瓷砖分类  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 15)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }
                std::string binary_dir = "/data/autostart/15/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_15_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                // cv::Rect top_rect_roi(RGB_image_src.cols * 0.3, RGB_image_src.rows * 0.5, RGB_image_src.cols * 0.5, RGB_image_src.rows * 0.3);
                // cv::Rect top_rect_roi(RGB_image_src.cols * 0.175, RGB_image_src.rows * 0.6, RGB_image_src.cols * 0.65, RGB_image_src.rows * 0.4);
                cv::Rect top_rect_roi(RGB_image_src.cols * 0.3, RGB_image_src.rows * 0.6, RGB_image_src.cols * 0.5, RGB_image_src.rows * 0.4);
                EcoDetectInference* ecoPureTileDetect_ = &ecocamtargetdetect_[12];  
                if (!ecoPureTileDetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoPureTileDetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }       
                // inference
                ecoestatus = ecoPureTileDetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoPureTileDetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 上镜头静态目标检测结果输出
                ecoCamOutputResult->ecogroundobjects = ecoPureTileDetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataiccls(15, ecoCamOutputResult, RGBDup, robot_lds_data, imu);

                if(-1 != access("/data/autostart/image/15/", 0))
                {
                    ////临时写个保存结果图片为了测试
                    ecoPureTileDetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/15/", ecoCamOutputResult->mask);
                }

                input_images.clear();
            }

        }
        else if (input_data.input_image[imgid].frame_id == 0)
        {///nogdc
/**********************************************************    人形目标识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 2)
            {
                std::vector<cv::Mat>            input_images;   
                std::vector<std::vector<float>> robot_lds_data(10, std::vector<float>(0, 0));
                std::vector<int>                imu;
                input_images.clear();
                for(int nvec = 0; nvec < 10; nvec++)
                {
                    robot_lds_data[nvec].clear();
                }
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                EcoDetectInference* ecoPersonDetect_ = &ecocamtargetdetect_[2]; 
                if (!ecoPersonDetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoPersonDetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }       
                // inference
                ecoestatus = ecoPersonDetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoPersonDetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }
                // 上镜头静态目标检测结果输出
                ecoCamOutputResult->ecogroundobjects = ecoPersonDetect_->getDetectObjects();

                // LDS 点云转图像坐标系
                std::vector<float> Tcl;
                lds2pixel(robot_lds_data, input_data.input_image[imgid].st.timestamp, 
                input_data.input_image[imgid].st, input_data.input_image[imgid].ldsPointsData, input_data.input_image[imgid].slPointsData,
                RGBDup.RGB, Tcl);

                ////下面这个函数中输入的1就是该模型对应的model_id,比如曾经的污渍检测对应的是23
                topcamstataicdectect(2, ecoCamOutputResult, RGBDup, robot_lds_data, imu);

                std::string binary_dir = "/data/autostart/2/";
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_.jpg";
                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                if(-1 != access("/data/autostart/image/2/", 0))
                {
                    ecoPersonDetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/2/", ecoCamOutputResult->mask);
                }
                input_images.clear();

            }

            /**********************************************************     颗粒物 + 污渍 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 7)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();


                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                std::string binary_dir = "/data/autostart/7/";
                 
                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_7_.jpg";

                    cv::imwrite(binary_path_SEG, RGB_image_src);
                }

                input_images.push_back(RGB_image_src);  

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 520, RGB_image_src.cols, RGB_image_src.rows - 520);

                EcoDetectInference* ecocamLiquiddetect_ = &ecocamtargetdetect_[6];
                if (!ecocamLiquiddetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamLiquiddetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                std::cout << "$$$$$$$$:modelSwitch: " << modelSwitch << std::endl;
                // inference  颗粒物开关二进制表示
                ecoestatus = ecocamLiquiddetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecocamLiquiddetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                ecoCamOutputResult->ecogroundobjects = ecocamLiquiddetect_->getDetectObjects();

                ////输入模型对应的model_id
                topcamstataicdectect(7, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/7/", 0))
                {
                    ecocamLiquiddetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/7/", ecoCamOutputResult->mask);
                }
                if (ecoCamOutputResult->ecogroundobjects->ngroundobjectnum > 0){
                    ecoCamOutputResult->mask = cv::Mat(384, 512,  CV_8UC3, cv::Scalar(114, 114, 114));
                    eco_resize(RGB_image_src, ecoCamOutputResult->mask, 512, 384, EM_RGA_NORMAL_RESIZE);
                }
                input_images.clear();
            }

        }
        else if(input_data.input_image[imgid].frame_id == 10)
        {
            /**********************************************************     红外图 AI 识别  ****************************************************************************************/
            if (input_data.input_image[imgid].model_id == 21)
            {
                std::vector<cv::Mat>            input_images;                // 模型输入图片
                std::vector<std::vector<float>> robot_lds_data;
                std::vector<int>                imu;
                input_images.clear();
                robot_lds_data.clear();
                imu.clear();

                //　输入的RGB图像数据数据转换
                cv::Mat RGB_image_src(input_data.input_image[imgid].image_rgb_height, input_data.input_image[imgid].image_rgb_width, CV_8UC3, (void *)input_data.input_image[imgid].image_rgb_data_addr);
                if(RGB_image_src.empty())
                {
                    ecoestatus = EStatus_OutOfMemory;
                    std::cout  <<  "RGB_image_src can't memory in model = " << input_data.input_image[imgid].model_id <<  ecoestatus << std::endl;
                    return ecoestatus;
                }

                input_images.push_back(RGB_image_src);  

                std::string binary_dir = "/data/autostart/21/";

                if (access(binary_dir.c_str(), 0) == 0) 
                {
                    cv::Mat RGB_image_src_clone = RGB_image_src.clone();
                    std::string binary_path_SEG = binary_dir + std::to_string(input_data.input_image[imgid].st.timestamp) + "_" + std::to_string(getTimeStamp()) + "_" + std::to_string(input_data.input_image[imgid].st.x) +
                    "_" + std::to_string(input_data.input_image[imgid].st.y) + "_" + std::to_string(input_data.input_image[imgid].st.Qz) + "_21_.jpg";
                    
                    cv::imwrite(binary_path_SEG, RGB_image_src_clone);
                }

                // 输入图片 ROI 用于二阶段
                cv::Rect top_rect_roi(0, 0, RGB_image_src.cols, RGB_image_src.rows);

                EcoDetectInference* ecoIRcamObjdetect_ = &ecocamtargetdetect_[13];
                if (!ecoIRcamObjdetect_->getOpenFlag())
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoIRcamObjdetect_->getOpenFlag() is false in EcoTaskInference::ecoTaskInfer" << std::endl;
                    return ecoestatus;
                }     

                ecoestatus = ecoIRcamObjdetect_->ecoDetectInfer(input_images, top_rect_roi, modelSwitch);
                if (ecoestatus)
                {
                    ecoestatus = EStatus_GenericError;
                    std::cout << "ecoIRcamObjdetect_->ecoDetectInfer(input_images, top_rect_roi) inference error" << ecoestatus << std::endl;
                    return ecoestatus;
                }

                // 颗粒物识别结果输出
                ecoCamOutputResult->ecogroundobjects = ecoIRcamObjdetect_->getDetectObjects();

                ////输入模型对应的model_id
                topircamdectect(21, ecoCamOutputResult, RGBDup, robot_lds_data, imu);
                if(-1 != access("/data/autostart/image/21/", 0))
                {
                    ecoIRcamObjdetect_->showDetectObjets(RGB_image_src, input_data.input_image[imgid].st, "/data/autostart/image/21/", ecoCamOutputResult->mask);
                }
                input_images.clear();

            }


        }


    }

    return EStatus_Success;
}


EcoEStatus EcoTaskInference::ecoTaskClose()
{
    EcoEStatus ecoestatus(EStatus_Success);
    if(!rug_mask.empty())
    {
        rug_mask.release();
    }

    // 释放镜头保存的结果
	if (NULL != ecocamoutputresult_)
	{
        for (size_t i = 0; i < nimgnum; i++)
        {
            ecocamoutputresult_[i].mask.release();
        }
        
        delete[] ecocamoutputresult_;
        ecocamoutputresult_= NULL;
	} 

    // 释放目标识别类
	if (NULL != ecocamtargetcls_)
	{
        for (size_t i = 0; i < nclsmodelnum; i++)
        {
            // 释放目标识别类
            ecoestatus = ecocamtargetcls_[i].ecoObjectClsClose();
            if (ecoestatus)
            {
                std::cout << "ecocamtargetcls_[ " << i << " ].ecoObjectClsClose() close error in EcoTaskInference::ecotaskclose" << ecoestatus << std::endl;
                return ecoestatus;
            }
        }

        delete[] ecocamtargetcls_;
        ecocamtargetcls_= NULL;
	}

    // 释放分割类
	if (NULL != ecocamtargetseg_)
	{
        for (size_t i = 0; i < nsegmodelnum; i++)
        {
            // 释放分割结果
            ecoestatus = ecocamtargetseg_[i].ecoSegClose();
            if (ecoestatus)
            {
                std::cout << "ecocamtargetseg_[ " << i << " ].ecoSegClose() close error in EcoTaskInference::ecotaskclose" << ecoestatus << std::endl;
                return ecoestatus;
            }
        }

        delete[] ecocamtargetseg_;
        ecocamtargetseg_= NULL;
	}

    // 释放目标检测类
	if (NULL != ecocamtargetdetect_)
	{
        for (int i = 0; i < ndetectmodelnum; i++)
        {
            // 释放目标检测结果
            ecoestatus = ecocamtargetdetect_[i].ecoDetectClose();
            if (ecoestatus)
            {
                std::cout << "ecocamtargetdetect_[ " << i << " ].ecoDetectClose() close error in EcoTaskInference::ecotaskclose" << ecoestatus << std::endl;
                return ecoestatus;
            }
        }

        delete[] ecocamtargetdetect_;
        ecocamtargetdetect_= NULL;
	}

    return EStatus_Success;
}


EcoEStatus EcoTaskInference::drawObjects(const cv::Mat& bgr, bool& save_image)
{
    EcoEStatus ecoestatus(EStatus_Success);
    
    if (bgr.empty())
    {
		ecoestatus = EStatus_InvalidParameter;
		std::cout  <<  "input img is empty:" <<  ecoestatus <<std::endl;
		return ecoestatus;
    }
    

    return EStatus_Success;
}


}



