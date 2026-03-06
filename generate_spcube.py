import yaml
from easydict import EasyDict
from datasets.kradar_detection_v1_0 import KRadarDetection_v1_0
import os

path_cfg = './configs/cfg_rl_3df_gate.yml'
f = open(path_cfg, 'r')
cfg = yaml.safe_load(f)
f.close()
cfg = EasyDict(cfg)

# Modify config for generation
cfg.DATASET.GET_ITEM['rdr_sparse_cube'] = False
cfg.DATASET.GET_ITEM['rdr_cube'] = True
cfg.DATASET.GET_ITEM['rdr_cube_doppler'] = False # No doppler data available
cfg.DATASET.RDR_CUBE.GENERATE_SPARSE_CUBE.IS_SAVE_TO_SAME_SEQUENCE = True
cfg.DATASET.RDR_CUBE.GENERATE_SPARSE_CUBE.NAME_SPARSE_CUBE = 'sparse_cube_gen'

print("Generating for train split...")
dataset_train = KRadarDetection_v1_0(cfg=cfg, split='train')
dataset_train.generate_sparse_rdr_cube()

print("Generating for test split...")
dataset_test = KRadarDetection_v1_0(cfg=cfg, split='test')
dataset_test.generate_sparse_rdr_cube()
