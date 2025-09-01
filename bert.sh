
. env.sh

export BWDHOME=$SEN_PROJECT_SRC
export FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS=10000
export FLEX_DEVICE=VFIO
export FLEX_COMPUTE=SENTIENT
export FORCE_XRF_ROWS=8
export SENPERFORMANCE=0
export DT_OPT=psum=0,lxopt=1,opfusion=1,arithfold=1,dataopt=1,weipreload=0,autopilot=0,dcc=1,dtversion=2
export PYTHONUNBUFFERED=1
export TORCH_SENDNN_TRAIN=1
export HF_HOME=$BWDHOME
export TORCH_LOGS=dynamo

stdbuf -i0 -o0 -e0 python3 train_classification_cpuopt.py --architecture=bert_classification --variant=base --unfuse_weights --num_classes=2 --checkpoint_format=hf --model_path=$BWDHOME/bert-base-uncased/model.safetensors  --tokenizer=$BWDHOME/bert-base-uncased --device_type=cpu --dataset_style=aml --dataset_path=./kyc_train_data_400.csv --compile --compile_backend="sendnn" --batch_size=4 --default_dtype="fp32" --epochs=1 --report_steps=1  2>&1 | tee ${0%.*}.log

./trainlog2csv.sh < ${0%.*}.log > ${0%.*}.csv

