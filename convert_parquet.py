"""Convert HuggingFace .parquet WikiArt dataset to flat image directory.

HuggingFace WikiArt datasets (e.g., huggan/wikiart) store images inside
.parquet files, often as bytes or PIL Image objects.  This script extracts
all images and saves them as .jpg files, making the dataset compatible
with train_microAST.py's FlatFolderDataset.

Usage:
    python convert_parquet.py --input_dir ./wikiart_parquet --output_dir ./wikiart/train
"""

import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True,
                    help='Directory containing .parquet files')
parser.add_argument('--output_dir', required=True,
                    help='Output directory for extracted .jpg images')
parser.add_argument('--image_column', default='image',
                    help='Column name containing image data '
                         '(common names: "image", "jpg", "bytes")')
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(input_dir.glob('*.parquet'))
assert parquet_files, f'No .parquet files found in {input_dir}'
print(f'Found {len(parquet_files)} parquet file(s)')

count = 0
for pf in tqdm(parquet_files, desc='Converting'):
    df = pd.read_parquet(pf)
    col = args.image_column

    # Auto-detect column name if user didn't specify the right one
    if col not in df.columns:
        for candidate in ['image', 'jpg', 'bytes', 'img', 'png']:
            if candidate in df.columns:
                col = candidate
                print(f'Auto-detected image column: "{col}"')
                break
        else:
            print(f'Columns in {pf.name}: {list(df.columns)}')
            raise KeyError(
                f'No image column found. Try --image_column with one of: '
                f'{list(df.columns)}')

    for _, row in df.iterrows():
        data = row[col]

        # Handle various storage formats used by HuggingFace
        if isinstance(data, bytes):
            img = Image.open(io.BytesIO(data))
        elif isinstance(data, Image.Image):
            img = data
        elif isinstance(data, dict) and 'bytes' in data:
            # HuggingFace Image feature: {'bytes': ..., 'path': ...}
            img = Image.open(io.BytesIO(data['bytes']))
        else:
            continue  # skip malformed rows

        img = img.convert('RGB')
        img.save(output_dir / f'{count:08d}.jpg', quality=95)
        count += 1

print(f'Extracted {count} images to {output_dir}/')

