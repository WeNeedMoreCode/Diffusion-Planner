# NOTE: device occupancy on this server is DYNAMIC (vllm/mindie containers drift).
# Check `npu-smi info` before each run and pick a device with <2GB usage (4-7 free at setup time).
export ASCEND_RT_VISIBLE_DEVICES=4
export HYDRA_FULL_ERROR=1
# Ray blanks accelerator-visibility env vars (ASCEND_RT_VISIBLE_DEVICES -> "") in workers
# when the task requests num_gpus=0 (our number_of_gpus_allocated_per_simulation=0).
# That makes torch.npu.is_available() False inside every Ray worker. Disable the override.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

###################################
# User Configuration Section
###################################
# Server paths (this NPU server; code synced from local Windows via Syncthing)
DP_ROOT="/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/Diffussion_Planner/Diffusion-Planner"
DP_DEVKIT="/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/Diffussion_Planner/nuplan-devkit"
# datasets/ckpt/exp live on /data (server root disk is chronically full; /data has 1.6T free)
DP_DATA="/data/syx_dp"

# Set environment variables
export NUPLAN_DEVKIT_ROOT="$DP_DEVKIT/"  # nuplan-devkit absolute path
export NUPLAN_DATA_ROOT="$DP_DATA/datasets/data/"  # nuplan dataset absolute path
export NUPLAN_MAPS_ROOT="$DP_DATA/datasets/maps/" # nuplan maps absolute path
export NUPLAN_EXP_ROOT="$DP_DATA/exp" # nuplan experiment absolute path

# Dataset split to use
# Options:
#   - "one_continuous_log"  (single log from the nuplan_mini db, quickest bring-up)
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="one_continuous_log"

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=diffusion_planner_release
ARGS_FILE="$DP_DATA/checkpoints/args.json"
CKPT_FILE="$DP_DATA/checkpoints/model.pth"

# val14 uses the full nuplan builder; everything else (mini db / one_continuous_log) uses nuplan_mini
if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_mini"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

# device is driven by config/planner/diffusion_planner.yaml (defaults to npu).
# NPU (Path C) notes vs the CUDA runner:
#   - ASCEND_RT_VISIBLE_DEVICES replaces CUDA_VISIBLE_DEVICES.
#   - number_of_gpus_allocated_per_simulation=0: nuPlan's initialize_ray() hardcodes the GPU count
#     from torch.cuda.device_count() (=0 on NPU) and the local ray.init() never passes num_gpus,
#     so Ray advertises 0 GPUs. A non-zero value here would leave sim tasks unscheduled (silent hang).
#   - Concurrency is instead capped by worker.threads_per_node (Ray CPU scheduling). Raise cautiously
#     to avoid piling too many sims onto one NPU (OOM).
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    scenario_filter.limit_total_scenarios=50 \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"
