#!/bin/bash
# build.debug.sh <cmake args ...>
#
# Boxxer example debug build script.
#
# Debug-only build to local install prefix with build-tree export

INSTALL_PATH=_install
BUILD_PATH=_build/Debug
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=OFF"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DOPT_DOC=Off"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=On"
ARGS="${ARGS} -DOPT_MATLAB=Off"

set -ex
#rm -rf $BUILD_PATH
cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -Wdev ${ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target all -- -j${NUM_PROCS}
#VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
