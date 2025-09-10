# parameter.py

# saving path
FOLDER_NAME = 'pred7_rdv'
model_path = f'checkpoints/{FOLDER_NAME}'
train_path = f'{model_path}/train'
gifs_path = f'{model_path}/gifs'

# predictor settings
generator_path = f'checkpoints/wgan_3000'
N_GEN_SAMPLE = 4
N_AGENTS = 3

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 500

# map and planning resolution
CELL_SIZE = 0.4                # meter
NODE_RESOLUTION = 4            # meter (graph step ~ 1 NODE_RESOLUTION)
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 16              # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 1

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 64
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 18

# network parameters
NODE_INPUT_DIM = 9
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25
NODE_PADDING_SIZE = 960

# GPU usage
USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 3

USE_WANDB = False

# ------------ Communication & travel cost ------------
COMMS_RANGE = 32.0
MAX_TRAVEL_COEF = 0
TOTAL_TRAVEL_COEF = 0

# ------------ RDV: selection & scheduling ------------
RDV_VERBOSE = True          # 打印流程日志
RDV_USE_ATTENTION_VIZ = False

# 可通行阈值（基于预测自由概率）
RDV_TAU_FREE = 0.6

# 候选池
RDV_ONLY_WHEN_FULLY_CONNECTED = True
RDV_SKIP_CAND_WHEN_ACTIVE     = True
RDV_CAND_UPDATE_EVERY         = 12
RDV_CAND_K                    = 400
RDV_CAND_STRIDE               = 4
RDV_TOP_M                     = 12
RDV_INFO_RADIUS_M             = 8.0

# 候选评分：IG - beta*ETA差异 - gamma*risk
RDV_ALPHA = 1.0
RDV_BETA  = 0.5
RDV_GAMMA = 0.2

# 窗口生成（围绕 max ETA）
RDV_WINDOW_ALPHA_EARLY = 0.20
RDV_WINDOW_BETA_EARLY  = 4.0
RDV_WINDOW_ALPHA_LATE  = 0.40
RDV_WINDOW_BETA_LATE   = 6.0

# 区域合同半径（相对通讯半径）
RDV_REGION_FRAC = 0.45

# 机会约束旅行时间
RDV_TT_N_SAMPLES = 24
RDV_EPSILON      = 0.2        # 1-eps 分位的保守旅行时间
RDV_MIN_LEAD_STEPS = 6        # 合同刚签后至少留这么多自由探索步

# ------------ RDV: execution ------------
# D* Lite 开销控制
RDV_PLAN_REPLAN_EVERY    = 4
RDV_COSTMAP_UPDATE_EVERY = 8
RDV_DSTAR_MAX_EXPAND     = 4000

# 进度/回退检查降频
RDV_PROGRESS_CHECK_EVERY = 5

# 成本图权重
RDV_RISK_LAMBDA = 3.0         # 风险权重
RDV_INFO_EPS    = 0.05        # 已知 free 的轻微惩罚，鼓励沿途探索

# 迟到容忍/回退策略
RDV_LATE_TOL            = 12   # 超窗宽限
RDV_FAIL_EXPAND_FACTOR  = 1.5  # 扩圈比例
RDV_SAFE_HUB_RADIUS     = 8.0  # 安全枢纽最小半径（米）
RDV_LATE_EXPAND_STEPS   = 60   # 若 ETA 很远，触发展开

# --------- Legacy rendezvous placeholders (unused) ---------
INTENT_HORIZON = 3
MEET_MODE = 'pred'
MEET_SYNC_TOL_M = 18.0
MEET_RADIUS_FRAC = 0.45
MEET_BUFFER_ALPHA = 0.2
MEET_BUFFER_BETA  = 4.0
MEET_LATE_TOL     = 12
R_MEET_SUCCESS    = 0
R_MEET_LATE       = 0
RENDEZVOUS_INFO_RADIUS_M = 8.0
