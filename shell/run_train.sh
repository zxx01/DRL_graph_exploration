source ~/anaconda3/bin/activate;
conda activate torch;

SCRIPT_PATH="$(readlink -f "$0")";
echo "Script directory is: ${SCRIPT_PATH}";
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_PATH")")";
echo "Script directory is: ${WORKSPACE_DIR}";
SCRIPT_PATH_DIR=${WORKSPACE_DIR}/scripts;
echo "Script directory is: ${SCRIPT_PATH_DIR}";

# export PYTHONPATH=${WORKSPACE_DIR}/build:$PYTHONPATH;
export PYTHONPATH=/home/zxx/workspace/ML_HW/DRL_graph_exploration/build:$PYTHONPATH;
export LIBRARY_PATH=/home/${USER}/anaconda3/envs/torch/lib/:$LIBRARY_PATH;
export LD_LIBRARY_PATH=/home/${USER}/anaconda3/envs/torch/lib/:$LD_LIBRARY_PATH;


python3 ${SCRIPT_PATH_DIR}/train.py

