
KB_DATA_PATH = 'scheduler_1_wzl/kb_data'

MODELS_PATH = 'scheduler_1_wzl/models'

NO_BAYES_GOAL = 0  # 按照遍历配置组合的方式来建立知识库
BEST_ALL_DELAY = 1  # 以最小化总时延为目标，基于贝叶斯优化建立知识库（密集，而不集中）
BEST_STD_DELAY = 2  # 以最小化不同配置间时延的差别为目标，基于贝叶斯优化建立知识库（稀疏，而均匀）

