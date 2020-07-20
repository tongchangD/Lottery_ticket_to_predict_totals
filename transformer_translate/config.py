import os
SOURCE_FILE="./data/in.txt"
TARGET_FILE="./data/out.txt"
ROOT_DIR="./data/data"
weights="./weights_i20200719"
if not os.path.exists(weights):
    os.makedirs(weights)
history=3
tag=False