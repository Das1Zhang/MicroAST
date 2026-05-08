# 该脚本用于下载数据并进行解压缩处理
set -e
pip install thop

pip install huggingface_hub

huggingface-cli login

hf download --repo-type dataset GAIA-URJC/COCO_2014 train2014.zip --local-dir .

hf download huggan/wikiart --repo-type dataset --include "data/*" --local-dir ./wikiart

mkdir -p coco2014/train2014
unzip -q train2014.zip -d coco2014/train2014


python convert_parquet.py --input_dir ./wikiar/parquet --output_dir ./wikiart/train

python train_microAST.py \
    --content_dir ./coco2014/train2014/train2014 \
    --style_dir ./wikiart/train \
    --save_dir ./exp \
    --checkpoints ./checkpoints \
    --log_dir ./logs \
    --sample_path ./samples \
    --n_threads 8 \
    --output ./output