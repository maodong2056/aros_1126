set -e

TARGET_SOC="rk356x"
GCC_COMPILER=/media/zhoufeng/zhoufeng_sda2/tools/RK3562-/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j4
make install
cd -