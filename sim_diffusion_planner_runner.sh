# Pick a free card with `npu-smi info` first (occupancy is dynamic on shared
# servers); DP_DEVICE overrides the default card 0 for multi-card machines.
# DP_WORKER=sequential runs single-process without Ray (troubleshooting mode;
# DP_THREADS is ignored there). DP_DEVICE / DP_LIMIT / DP_THREADS override
# for benchmark runs.
export ASCEND_RT_VISIBLE_DEVICES=${DP_DEVICE:-0}
export HYDRA_FULL_ERROR=1
# Ray blanks accelerator-visibility env vars (ASCEND_RT_VISIBLE_DEVICES -> "") in workers
# when the task requests num_gpus=0 (our number_of_gpus_allocated_per_simulation=0).
# That makes torch.npu.is_available() False inside every Ray worker. Disable the override.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

###################################
# User Configuration Section
###################################
# Paths are resolved relative to this script (portable across machines):
#   datasets/  checkpoints/  exp/  torchair_cache/ live inside the project dir.
# Layout assumption: nuplan-devkit is a sibling checkout (see README).
# Override DP_DATA (e.g. export DP_DATA=/data/syx_dp) when the data lives on
# another disk; DP_DEVICE picks the NPU card (check npu-smi info first).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DP_ROOT="$SCRIPT_DIR"
DP_DEVKIT="$(dirname "$SCRIPT_DIR")/nuplan-devkit"
DP_DATA=${DP_DATA:-"$SCRIPT_DIR"}
export DP_TORCHAIR_CACHE=${DP_TORCHAIR_CACHE:-"$DP_DATA/torchair_cache"}

# Set environment variables
export NUPLAN_DEVKIT_ROOT="$DP_DEVKIT/"  # nuplan-devkit absolute path
# Official mini archives extract straight to <dir>/mini/ (no nuplan-v1.1/splits
# prefix), so the runner overrides scenario_builder.data_root to point at
# cache/mini directly instead of relying on NUPLAN_DATA_ROOT + the hardcoded
# suffix in nuplan_mini.yaml (see DB_ROOT_OVERRIDE below).
export NUPLAN_DATA_ROOT="$DP_DATA/datasets/data/cache/"  # nuplan dataset absolute path
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
DB_ROOT_OVERRIDE=""
if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_mini"
    # archives extract to cache/mini/ -- bypass the yaml's nuplan-v1.1/splits suffix
    DB_ROOT_OVERRIDE="scenario_builder.data_root=$DP_DATA/datasets/data/cache/mini"
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
    $DB_ROOT_OVERRIDE \
    scenario_filter=$SPLIT \
    scenario_filter.limit_total_scenarios=${DP_LIMIT:-50} \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=${DP_WORKER:-ray_distributed} \
    worker.threads_per_node=${DP_THREADS:-4} \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"
