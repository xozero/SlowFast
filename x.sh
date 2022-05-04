python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR datasets/dataset1/splits \
  DATA.PATH_PREFIX datasets/dataset1/data/x1 \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 16 \
