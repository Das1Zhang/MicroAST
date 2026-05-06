set -e
pip install huggingface_hub

huggingface-cli login

hf download --repo-type dataset GAIA-URJC/COCO_2014 train2014.zip --local-dir .

hf download huggan/wikiart --repo-type dataset --include "data/*" --local-dir .

mkdir -p coco2014/train2014
unzip -q train2014.zip -d coco2014/train2014


