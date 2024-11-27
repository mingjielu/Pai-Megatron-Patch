# cd /workspace/Pai-Megatron-Patch/examples/mistral

MBS=2
GBS=256
TP=4
PP=1
EP=1
GROUP_GEMM=false  # true
AC=none
DO=true
FL=true # FA
SP=true
TE=true
MOE=true
SEQ_LEN=4096
PAD_LEN=4096
MODEL_SIZE=7B
PR=fp8 # fp8
OPT=false
EXP_DIR=${PWD}/experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG=${EXP_DIR}/MI308-Mixtral8x${MODEL_SIZE}-${PR}-seq${SEQ_LEN}-te_${TE}-fa_${FL}-tp${TP}pp${PP}ep${EP}-mbs${MBS}-gbs${GBS}-opt_${OPT}-${TIMESTAMP}.log

if [ "$OPT" = "true" ]
then
    echo "Use Optimization for Bubble Reduction..."
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    # export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    # export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=0
    # export NCCL_SOCKET_IFNAME=ens51f0np0
    # export GLOO_SOCKET_IFNAME=ens51f0np0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    # export AMD_LOG_LEVEL=3
    # export AMD_SERIALIZE_KERNEL=3
    export HSA_NO_SCRATCH_RECLAIM=1
    # export RCCL_MSCCLPP_ENABLE=0
    # export HSA_ENABLE_IPC_MODE_LEGACY=1
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10 # can be higher value
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=50 # can be higher value
fi

sh run_finetune_mcore_mistral_withGA_AMD.sh  \
dsw  \
../../ \
${MODEL_SIZE}   \
${MBS}    \
${GBS} \
1e-5   \
1e-6   \
${SEQ_LEN}  \
${PAD_LEN}  \
0   \
${PR}  \
${TP}   \
${PP}  \
${AC}  \
${DO}   \
${FL}  \
${SP}   \
${TE}   \
${MOE} \
100000  \
${EXP_DIR}/mistral-datasets/alpaca_zh-mistral-train.json   \
${EXP_DIR}/mistral-datasets/alpaca_zh-mistral-valid.json   \
${EXP_DIR}/model_utils/tokenizer_7b   \
10   \
2   \
${EXP_DIR}/output_mcore_mistral   \
${EP}   \
${GROUP_GEMM} \
2>&1 | tee ${TRAIN_LOG}


echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[5:]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(f"{mean:.2f}")' > mean_log_value.py


echo '=========================='
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
echo "throughput per GPU: $(python mean_log_value.py tmp.txt)"
rm tmp.txt

echo '=========================='
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
echo "elapsed time per iteration: $(python mean_log_value.py tmp.txt)"
rm tmp.txt

echo '=========================='
grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
echo "mem usages: $(python mean_log_value.py tmp.txt)"
rm tmp.txt
