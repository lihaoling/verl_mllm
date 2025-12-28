set -x

# export http_proxy=http://star-proxy.oa.com:3128
# export https_proxy=http://star-proxy.oa.com:3128

unset http_proxy
unset https_proxy

# 确保使用本地 verl_mllm 代码（优先级高于 /root/verl）
export PYTHONPATH="/apdcephfs/private_ringohlli_qy4/project/verl_mllm:$PYTHONPATH"

# 禁用 InfiniBand，强制使用 Socket 通信（跨网段场景）
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# 指定网卡
export NCCL_SOCKET_IFNAME=bond1

# 增加超时和重试
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Socket 通信优化
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

export HYDRA_FULL_ERROR=1

# 修复 PyTorch CUDA 编译时无法检测架构的问题
# 8.0 = A100, 8.9 = H100, 9.0 = H100 (Hopper)
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
export VLLM_USE_V1=1

export SANDBOX_FUSION_ENDPOINT="http://localhost:8080"
export WANDB_API_KEY="fc7022e7e115dbc7bef672a9137ebb0618ec9160"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/apdcephfs/private_ringohlli_qy4/project/verl_mllm/data/geo3k/train.parquet \
    data.val_files=/apdcephfs/private_ringohlli_qy4/project/verl_mllm/data/geo3k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/apdcephfs_zwfy/share_304017095/ringohlli/model/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    global_profiler.tool=torch_memory \
    global_profiler.save_path=./mem_snapshots \
    global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=100000 \
    global_profiler.global_tool_config.torch_memory.stack_depth=32 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.mode=async \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='mllm_verl_grpo_exp' \
    trainer.experiment_name='qwen25_vl_7b_line22_geo3k_grpo_1e6_bsz512_2048_8192_1226' \
    trainer.default_local_dir=/apdcephfs_zwfy/share_304017095/ringohlli/ckpt/grpo/qwen25_vl_7b_line22_geo3k_grpo_1e6_bsz512_2048_8192_1226 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15


# pip install -e . --break-system-packages
# apt install tmux -y
# ray start --head --num-gpus=8 --port=6380 --include-dashboard=false --block  &




# # 修复pssh命令格式
# pssh -H "29.81.228.18  29.81.226.211  29.79.9.237  29.81.226.148  29.81.242.44  29.81.242.150  29.79.9.239  29.81.240.235  29.81.242.47  29.81.244.43  29.81.228.152  29.79.9.158  29.79.8.108  29.81.240.240  29.81.244.116" -i "
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_SL=3
# export NCCL_CHECK_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=0
# export NCCL_LL_THRESHOLD=16384
# export NCCL_IB_CUDA_SUPPORT=1

# export NCCL_SOCKET_IFNAME=bond1
# export UCX_NET_DEVICES=bond1
# export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6

# export NCCL_COLLNET_ENABLE=0
# export SHARP_COLL_ENABLE_SAT=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_PXN_DISABLE=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_TIMEOUT=1800

# export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
# export VLLM_USE_V1=1

# export SANDBOX_FUSION_ENDPOINT="http://localhost:8080"
# export WANDB_API_KEY="fc7022e7e115dbc7bef672a9137ebb0618ec9160"
# unset http_proxy
# unset https_proxy
# nohup ray start --address=28.58.224.68:6380 --num-gpus=8 --block > /dev/null 2>&1 &
# "




# pssh -i -t 0 -h pssh.hosts -i "
# export http_proxy=http://star-proxy.oa.com:3128
# export https_proxy=http://star-proxy.oa.com:3128
# git clone https://github.com/lihaoling/verl_mllm.git
# cd verl_mllm
# pip install -e .
# "


