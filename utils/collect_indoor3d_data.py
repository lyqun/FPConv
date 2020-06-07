"""
    https://github.com/charlesq34/pointnet/blob/master/sem_seg/collect_indoor3d_data.py
"""
import os, sys
import indoor3d_util
import argparse
import json

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--config", type=str, default='../config.json')
args = parser.parse_args()
with open(args.config, 'r') as f:
    _cfg = json.load(f)
    print(_cfg)

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
META_ROOT = os.path.join(UTILS_DIR, 's3dis_meta')

RAW_ROOT = _cfg['s3dis_aligned_raw']
OUTPUT_ROOT = _cfg['s3dis_data_root']


print('meta root:', META_ROOT)
print('raw root:', RAW_ROOT)
print('output root:', OUTPUT_ROOT)
anno_paths = os.path.join(META_ROOT, 'annotations.txt')
anno_paths = [line.rstrip() for line in open(anno_paths)]

output_folder = OUTPUT_ROOT
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# NOTE: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    elements = anno_path.split('/')
    out_filename = elements[-3] + '_' + \
        elements[-2] + '.npy'
    if os.path.exists(os.path.join(output_folder, out_filename)):
        continue
    indoor3d_util.collect_point_label(os.path.join(RAW_ROOT, anno_path), 
                                      os.path.join(output_folder, out_filename), 
                                      'numpy', False)
