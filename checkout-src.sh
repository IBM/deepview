#!/bin/bash

# exit on error
set -o errexit -o nounset -o pipefail

_SRC=${1:-aiu-src}
_TAG=${2:-${AI_CHIP_TOOLSCHAIN_TAG:-}}

mkdir -p "$_SRC"
cd "$_SRC"

function checkout_repo() {
    _NAME=$1
    if [ -d "$_NAME" ]; then
            rm -rf ${_NAME}
    fi
    echo "===================== "${_NAME}
    git clone -q --recurse-submodules git@github.ibm.com:IBM/${_NAME}.git

    case "$_NAME" in
        aiu-toolbox)    tag=${AIU_TOOLBOX_TAG:-$_TAG} ;;
        dee)            tag=${DEE_TAG:-$_TAG} ;;
        deeptools)      tag=${DEEPTOOLS_TAG:-$_TAG} ;;
        flex)           tag=${FLEX_TAG:-$_TAG} ;;
        senbfcc)        tag=${SENBFCC_TAG:-$_TAG} ;;
        sendnn)         tag=${SENDNN_TAG:-$_TAG} ;;
        torch_sendnn)   tag=${TORCH_SENDNN_TAG:-$_TAG} ;;
    esac

    if [ -n "$tag" ]; then
        (cd "$_NAME" && git checkout -q "$tag")
    fi
}

checkout_repo aiu-toolbox
checkout_repo dee
checkout_repo deeptools
checkout_repo flex
checkout_repo senbfcc
checkout_repo sendnn
checkout_repo torch_sendnn

if [[ "${MANAGE_LLVM:-}" != on ]]; then
    echo "===================== llvm-project"
    rm -rf llvm-project
    git clone https://github.com/llvm/llvm-project.git

    # LLVM commit hash is defined here: https://github.ibm.com/IBM/deeptools/blob/stable_2025_04_16/cmake/external_mlir.cmake#L24
    tag=${LLVM_TAG:-dd6f6a096a59892ce1f9c454461aa5ed4c2aa971}
    if [ -n "$tag" ]; then
        (cd llvm-project && git checkout -q "$tag")
    fi
fi

if [[ $(uname -m) != ppc64le ]]; then
    echo "===================== aiu-fms-testing-utils"
    rm -rf aiu-fms-testing-utils
    git clone -q https://github.com/foundation-model-stack/aiu-fms-testing-utils.git
    if [ -n "${AIU_FMS_TAG:-}" ]; then
        (cd aiu-fms-testing-utils && git checkout -q "$AIU_FMS_TAG")
    fi

    echo "===================== foundation-model-stack"
    # TODO: We temporarily fork the foundation-model-stack repo to apply our own patches for backward kernels.
    # We will switch back to use the upstream repo when the issues are fixed.
    # git clone -q https://github.com/foundation-model-stack/foundation-model-stack.git
    rm -rf foundation-model-stack
    git clone -q git@github.ibm.com:JL25131/foundation-model-stack.git
    if [ -n "${FMS_TAG:-}" ]; then
        (cd foundation-model-stack && git checkout -q "$FMS_TAG")
    fi
fi
rm -rf build
