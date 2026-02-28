cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)


# rknn api
set(CMAKE_CONFIG_PATH /home/xuan/project/ros_rknn_3562_sweeper_zj2425/config/ai_params.json)
set(CMAKE_AI_CONFIG_PATH /home/xuan/project/ros_rknn_3562_sweeper_zj2425/config/new_test_json_eight.json)
# set(CMAKE_IMAGE_PATH  /media/zhoufeng/zhoufeng_sda2/deboot_sweeper_ai/ros_rknn_3562_sweeper_dl_zj2435/img)
set(CMAKE_MODEL_PATH  /home/xuan/project/ros_rknn_3562_sweeper_zj2425)
# set(RKNN_SHARE_PATH   /home/wang/code/rk_dibao/rknn-toolkit2-master/rknpu2)
set(RKNN_SHARE_PATH   /mnt/7B5E2C3F52A49663/eco/rknn-toolkit2-master/rknpu2)

# if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  # add_definitions(-DD_DEBUG)
# endif()
# add_definitions(-DD_DPOSTEBUG)
# add_definitions(-DD_DEBUG_SAVETXT)
# add_definitions(-DD_DEBUG)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined,-ldl,-g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined,-ldl,-g")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ${RKNN_SHARE_PATH}/runtime/${CMAKE_SYSTEM_NAME}/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${RKNN_SHARE_PATH}/runtime/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  if (CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)

# opencv
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${RKNN_SHARE_PATH}/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ${RKNN_SHARE_PATH}/examples/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    # set(OpenCV_DIR ${RKNN_SHARE_PATH}/examples/3rdparty/opencv/opencv-linux-aarch64/lib/cmake/opencv4)
    # set(OpenCV_DIR ${RKNN_SHARE_PATH}/examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
    set(OpenCV_DIR /mnt/7B5E2C3F52A49663/eco/libopencv_4.1.2/libopencv-t10-px3/lib/cmake/opencv4)
  endif()
endif()

# rga
# comes from https://github.com/airockchip/librga
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RGA_LIB ${RKNN_SHARE_PATH}/examples/3rdparty/rga/libs/AndroidNdk/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
else()
  if(CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()

  set(RGA_LIB ${RKNN_SHARE_PATH}/examples/3rdparty/rga/libs/Linux/gcc-${LIB_ARCH}/librga.so)
endif()

find_package(OpenCV REQUIRED)
message("OpenCV version  is : ${OpenCV_VERSION}")
message("RKNN_API_PATH = ${RKNN_API_PATH}")
message("OpenCV_DIR    = ${OpenCV_DIR}")

# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)
include_directories( ${CMAKE_SOURCE_DIR}/include/rapidjson)
include_directories( ${CMAKE_SOURCE_DIR}/include/rga)
include_directories( ${OpenCV_INCLUDE_DIRS})

# set(paht opencv_lib /media/zhoufeng/zhoufeng_sda2/tools/RK3588/rknpu2_1.3.0.tar/rknpu2_1.3.0/rknpu2_1.3.0/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a)

file(GLOB_RECURSE FILE_SOURSE_LIST ${CMAKE_SOURCE_DIR}/src/*.cc)
# file(GLOB_RECURSE OpenCV_LIBS ${opencv_lib}/*.a)


message("1.OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message("2.OpenCV_LIBS = ${OpenCV_LIBS}")
message("3.FILE_SOURSE_LIST = ${FILE_SOURSE_LIST}")


add_library(ecoaisweeper SHARED ${FILE_SOURSE_LIST})
# target_link_libraries(ecoaisweeper  ${RKNN_RT_LIB} ${OpenCV_LIBS})
target_link_libraries(ecoaisweeper  ${RKNN_RT_LIB} ${OpenCV_LIBS} ${RGA_LIB})

add_library(ecopersonremove SHARED ${CMAKE_SOURCE_DIR}/srcperson/person_remove.cc)
target_link_libraries(ecopersonremove  ${OpenCV_LIBS} dl)


add_executable(rknn_detect_demo
  test/detect_self.cc
)

add_executable(rknn_seg_demo
  test/seg.cc
)

add_executable(rknn_infer_demo
  test/infer.cc
)

add_executable(rknn_main_task_demo
  test/main_task_self.cc
)

target_link_libraries(rknn_main_task_demo
  ecoaisweeper
  dl
)

target_link_libraries(rknn_detect_demo
  ecoaisweeper
)

target_link_libraries(rknn_seg_demo
  ecoaisweeper
)

target_link_libraries(rknn_infer_demo
  ecoaisweeper
)

install(TARGETS rknn_main_task_demo           DESTINATION ./)
install(TARGETS rknn_detect_demo              DESTINATION ./)
install(TARGETS rknn_seg_demo                 DESTINATION ./)
install(TARGETS rknn_infer_demo               DESTINATION ./)

# install(DIRECTORY  ${CMAKE_IMAGE_PATH}        DESTINATION ./)
# install(DIRECTORY  ${CMAKE_IMAGE_PATH}        DESTINATION ./)
# install(DIRECTORY  ${CMAKE_MODEL_PATH}/model  DESTINATION ./)

install(PROGRAMS  ${RKNN_RT_LIB}                           DESTINATION ./lib)
install(TARGETS   ecoaisweeper                             DESTINATION ./lib)
install(PROGRAMS  ${RGA_LIB}                               DESTINATION ./lib)
install(TARGETS   ecopersonremove                          DESTINATION ./lib)
install(DIRECTORY ${CMAKE_MODEL_PATH}/model/EcoAiSweeper   DESTINATION ./model/)
install(PROGRAMS  ${CMAKE_AI_CONFIG_PATH}                  DESTINATION ./model/)


install(TARGETS   ecoaisweeper                             DESTINATION ${CMAKE_SOURCE_DIR}/Ai_sweeper/root/usr/lib/eco_ai_interface)
install(PROGRAMS  ${RKNN_RT_LIB}                           DESTINATION ${CMAKE_SOURCE_DIR}/Ai_sweeper/root/usr/lib)
install(DIRECTORY ${CMAKE_MODEL_PATH}/model/EcoAiSweeper   DESTINATION ${CMAKE_SOURCE_DIR}/Ai_sweeper/root/model/)
install(TARGETS   ecopersonremove                          DESTINATION ${CMAKE_SOURCE_DIR}/Ai_sweeper/root/usr/lib)
install(PROGRAMS  ${CMAKE_CONFIG_PATH}                     DESTINATION ${CMAKE_SOURCE_DIR}/Ai_sweeper/root/etc/conf)





