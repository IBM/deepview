pipeline {
    agent {
        node {
            label "${params.NODE_LABEL}"
        }
    }
    environment {
        OC_LOGIN_CREDS = credentials('137d090f-5d8a-49e4-b175-094aea39a628')
        WXPE_GH_CREDS = credentials('aiu-gh-read-pat')
        HF_TOKEN = credentials('6f916cb2-6728-4d86-b560-5d73f20218a1')
        SPYRE_SERVICE_ID_TOKEN = credentials('wxpe-service-id-token')
    }

    parameters {
        string(name: 'CLUSTER_URL', defaultValue: 'https://api.dev.spyre.res.ibm.com:6443', description: '')
        string(name: 'IMAGE_URL', defaultValue: 'icr.io/ibmaiu_internal/x86_64/dd2/e2e_stable:latest', description: '')
        string(name: 'COMPONENT_NAME', defaultValue: 'deepview')
        string(name: 'AIU_RESOURCE_REQUEST', defaultValue: "1", description: '')
        string(name: 'CPU_REQUEST', defaultValue: "4", description: '')
        string(name: 'MEMORY_REQUEST', defaultValue: "8Gi", description: '')
        string(name: 'AIU_RESOURCE_LIMITS', defaultValue: "1", description: '')
        string(name: 'CPU_LIMITS', defaultValue: "32", description: '')
        string(name: 'MEMORY_LIMITS', defaultValue: "256Gi", description: '')
        string(name: 'IS_MULTI_AIU', defaultValue: "0", description: '')
        string(name: 'FMS_GIT_COMMIT', defaultValue: '', description: 'if left blank then it wont install this module. connected to SETUP_FMS_TEST_ENV variable so when checked it looks for this variable')
        string(name: 'AIU_FMS_TESTING_UTILS_GIT_COMMIT', defaultValue: 'main', description: 'if left blank then it wont install this module. connected to SETUP_FMS_TEST_ENV variable so when checked it looks for this variable')
        string(name: 'AIU_TESTS_GIT_COMMIT', defaultValue: 'main', description: 'if left blank then it wont install this module. connected to SETUP_FMS_TEST_ENV variable so when checked it looks for this variable')
        string(name: 'HF_HOME', defaultValue: '/home/senuser/models/huggingface_cache/hub', description: 'set /home/senuser/models/hf_cache for AIS-5')
        string(name: 'HF_HUB_OFFLINE', defaultValue: '', description: 'set 1 to use already downloaded models')
        string(name: 'HF_HUB_CACHE', defaultValue: "/home/senuser/models/huggingface_cache/hub", description: 'set 1 to use already downloaded models')
        string(name: 'TEST_MODEL_DIR', defaultValue: '/home/senuser/models')
        string(name: 'TEST_MODEL_PVC', defaultValue: 'shared-models')
        string(name: 'FMS_TEST_SHAPES_USE_MICRO_MODELS', defaultValue: '0', description: 'Set 1 for tiny model')
        string(name: 'FMS_TEST_SHAPES_MICRO_MODELS_HOME', defaultValue: "/home/senuser/models/tiny-models", description: 'This path is required when FMS_TEST_SHAPES_USE_MICRO_MODELS is set 1')
        string(name: 'CLUSTER_NAMESPACE', defaultValue: "a5-deepview", description: '')
        string(name: 'OMP_NUM_THREADS', defaultValue: "", description: 'Use if less resources are vailable')
        string(name: 'TORCHINDUCTOR_COMPILE_THREADS', defaultValue: "", description: 'Use if less resources are vailable')
        string(name: 'DT_PARALLEL_THREADS', defaultValue: "", description: 'Use if less resources are vailable')
        booleanParam(defaultValue: false, description: 'use this for development, it wont use dedicated cicd resources', name: 'IS_DEV_POD')
        booleanParam(defaultValue: false, description: 'when checked, it will install all commints added in FMS_GIT_COMMIT,AIU_FMS_TESTING_UTILS_GIT_COMMIT, AIU_TESTS_GIT_COMMIT etc ', name: 'SETUP_FMS_ENV')
        booleanParam(defaultValue: false, description: 'select this if you want to install aftu in vllm image. leave false as vllm image already has aftu', name: 'INSTALL_AFTU_NO_DEPS')
        booleanParam(defaultValue: false, description: 'use this when using vllm image, it will clone and put aftu in pod', name: 'SETUP_FMS_TEST_ENV')
        booleanParam(defaultValue: false, description: 'set true when when using spyre/e2e_stable images ', name: 'APPLY_DOOM_FIX')
        booleanParam(defaultValue: true, description: 'set true to delete pod after completion', name: 'CLEANUP_POD') 
    }

    stages {
        stage('Checkout') {
            steps {
                sh '''#!/bin/bash
                    git clone "https://${WXPE_GH_CREDS_USR}:${WXPE_GH_CREDS_PSW}@github.ibm.com/ai-foundation/aiu-tests" 
                    cd aiu-tests && git checkout ${AIU_TESTS_GIT_COMMIT} && git pull --all && cd ../

                    # install the following when using taas kubernetes node
                    if [[ "$NODE_LABELS" == *"taas_image"* ]]; then
                        sudo apt-get update 
                        sudo apt-get -y install gettext
                    fi
                    git clone "https://${WXPE_GH_CREDS_USR}:${WXPE_GH_CREDS_PSW}@github.com/IBM/deepview" 
                '''
            }
        } 
        stage('oc login') {
            steps {
                retry(count: 3){
                    sh '''#!/bin/bash
                        oc login --server=${CLUSTER_URL} --token ${SPYRE_SERVICE_ID_TOKEN} -n a5-deepview
                    '''
                }
            }
        }
        stage('create test pod') {
            steps {
                sh 'printenv'
                sh '''#!/bin/bash
                    aiu-tests/scripts/gen_env.sh
                    source aiu-tests/scripts/env.sh
                    aiu-tests/scripts/render.sh
                    cat ${POD_YAML}
                    oc apply -f ${POD_YAML} -n ${CLUSTER_NAMESPACE} && oc wait --for=condition=Ready --timeout=200s pod/${POD_NAME} -n ${CLUSTER_NAMESPACE}
                    oc describe pod ${POD_NAME} -n ${CLUSTER_NAMESPACE}
                    oc rsync --no-perms --container app --quiet deepview/ ${POD_NAME}:/home/senuser/deepview -n ${CLUSTER_NAMESPACE}
                    if [[ "$APPLY_DOOM_FIX" == "true" ]]; then
                            # fix for DOOM error
                            oc rsync --no-perms --container app --quiet aiu-tests/scripts/dd2/ ${POD_NAME}:/home/senuser -n ${CLUSTER_NAMESPACE}
                    fi
                    if [[ "${SETUP_FMS_ENV}" == "true" ]]; then
                        echo "setting up fms env"
                        if [[ "${IMAGE_URL}" == *"vllm"* ]]; then
                            echo "using vllm image, skipping fms install"
                            oc exec --container app -n ${CLUSTER_NAMESPACE} -i ${POD_NAME} -- bash -lc "aiu-tests/scripts/entry.sh"
                        else
                            oc exec --container app -n ${CLUSTER_NAMESPACE} -i ${POD_NAME} -- bash -lc "pip3 install pytest-forked && pip3 install datasets && aiu-tests/scripts/entry.sh"
                        fi
                    fi
                    if [[ "${SETUP_FMS_TEST_ENV}" == "true" ]]; then
                        git clone "https://github.com/foundation-model-stack/aiu-fms-testing-utils" 
                        cd aiu-fms-testing-utils && git checkout ${AIU_FMS_TESTING_UTILS_GIT_COMMIT} && git pull --all && cd ../
                        oc rsync --no-perms --container app --quiet aiu-fms-testing-utils/ ${POD_NAME}:/home/senuser/aiu-fms-testing-utils -n ${CLUSTER_NAMESPACE}
                    fi
            '''
            }
        }
        stage('setup dev deps') {
            steps {
                sh '''#!/bin/bash
                    oc exec --container app -n ${CLUSTER_NAMESPACE} -i ${POD_NAME} -- bash -lc "cd deepview && pip3 install -e .&& pip3 install .[dev]"
                '''
            }
        }
        stage('format and lint') {
            steps {
                sh '''#!/bin/bash
                    oc exec --container app -n ${CLUSTER_NAMESPACE} -i ${POD_NAME} -- bash -lc "cd deepview && tox -e ruff"
                '''
            }
        }
        stage('execute tests') { 
            steps {
                sh '''#!/bin/bash
                    oc exec --container app -n ${CLUSTER_NAMESPACE} -i ${POD_NAME} -- bash -lc "cd deepview && pytest"
                '''
            }
        }
    }

    post {
        always {
            script {
                if (params.CLEANUP_POD) {
                    try{
                        sh '''#!/bin/bash
                            oc delete -f ${POD_YAML} -n ${CLUSTER_NAMESPACE}
                        '''  
                    } catch (Exception e) {
                        echo "Error occured while clean up: "+ e.toString()
                    } 
                }
            }
        }
        cleanup {
            cleanWs()
        }
    }
}