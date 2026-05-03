"""Video style transfer with temporal consistency.

Processes a directory of video frames through MicroAST with EMA-based
temporal smoothing to suppress flickering between consecutive frames.

Usage:
    # Process all frames in a directory with a single style image
    python test_video_microAST.py \\
        --content_dir path/to/frames/ \\
        --style path/to/style.jpg \\
        --output path/to/output/

    # Adjust smoothing strength (0.5 = light, 0.9 = heavy)
    python test_video_microAST.py \\
        --content_dir path/to/frames/ \\
        --style path/to/style.jpg \\
        --temporal_momentum 0.8

The script processes frames in alphanumeric sorted order and applies
EMA smoothing to:
  1. Content encoder features — primary flicker source
  2. Style modulation signals — for changing-style scenarios
  3. Modulator outputs (w, b) — numerical stability

Scene cuts: if your video contains scene cuts, the smoother will bleed
signals across the cut. Either process each shot separately or use the
--scene_cut_frames argument to reset the smoother at specific frames.
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import net_microAST as net


def test_transform(size, crop):
    """Build the image preprocessing transform."""
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


parser = argparse.ArgumentParser(
    description='Video style transfer with temporal consistency.')

# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory containing video frames (processed in '
                         'alphanumeric sorted order)')
parser.add_argument('--style', type=str,
                    help='File path to the style image')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to style images '
                         '(one style per frame, must match frame count)')

# Model checkpoints
parser.add_argument('--content_encoder', type=str,
                    default='models/content_encoder_iter_160000.pth.tar')
parser.add_argument('--style_encoder', type=str,
                    default='models/style_encoder_iter_160000.pth.tar')
parser.add_argument('--modulator', type=str,
                    default='models/modulator_iter_160000.pth.tar')
parser.add_argument('--decoder', type=str,
                    default='models/decoder_iter_160000.pth.tar')

# Transform options
parser.add_argument('--content_size', type=int, default=0,
                    help='Resize content frames to this size '
                         '(0 = keep original)')
parser.add_argument('--style_size', type=int, default=0,
                    help='Resize style image to this size '
                         '(0 = keep original)')
parser.add_argument('--crop', action='store_true',
                    help='Center-crop to create square images')

# Output options
parser.add_argument('--output', type=str, default='output_video',
                    help='Directory to save output frames')
parser.add_argument('--save_ext', default='.jpg',
                    help='Output image format extension')

# Temporal smoothing options
parser.add_argument('--temporal_momentum', type=float, default=0.7,
                    help='EMA momentum for temporal smoothing [0, 1]. '
                         'Higher = more temporal consistency but slower '
                         'response to motion. 0.0 = no smoothing. '
                         'Recommended: 0.5-0.9 (default: 0.7)')
parser.add_argument('--scene_cut_frames', type=str, default=None,
                    help='Comma-separated list of frame indices where '
                         'scene cuts occur. The smoother is reset after '
                         'each cut. E.g., "0,150,300"')

# Style options
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Stylization strength [0, 1]')
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
device = torch.device('cuda:%d' % args.gpu_id)

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Collect content frames (sorted for temporal order)
content_dir = Path(args.content_dir)
content_paths = sorted(content_dir.glob('*'),
                       key=lambda p: (p.stem.zfill(10), p.suffix))
assert len(content_paths) > 0, f"No frames found in {args.content_dir}"
print(f"Found {len(content_paths)} frames in {args.content_dir}")

# Collect style images
if args.style:
    style_paths = [Path(args.style)] * len(content_paths)
elif args.style_dir:
    style_dir = Path(args.style_dir)
    style_paths = sorted(style_dir.glob('*'),
                         key=lambda p: (p.stem.zfill(10), p.suffix))
    if len(style_paths) == 1:
        style_paths = style_paths * len(content_paths)
    elif len(style_paths) != len(content_paths):
        raise ValueError(
            f"Style dir has {len(style_paths)} images but content has "
            f"{len(content_paths)} frames. They must match or style dir "
            f"must have exactly 1 image.")
else:
    raise ValueError("Either --style or --style_dir must be provided.")

# Parse scene cuts
scene_cuts = set()
if args.scene_cut_frames:
    scene_cuts = {int(x.strip()) for x in args.scene_cut_frames.split(',')}

# ------------------------------------------------------------------
# Build model
# ------------------------------------------------------------------
content_encoder = net.Encoder()
style_encoder = net.Encoder()
modulator = net.Modulator()
decoder = net.Decoder()

content_encoder.eval()
style_encoder.eval()
modulator.eval()
decoder.eval()

content_encoder.load_state_dict(torch.load(args.content_encoder))
style_encoder.load_state_dict(torch.load(args.style_encoder))
modulator.load_state_dict(torch.load(args.modulator))
decoder.load_state_dict(torch.load(args.decoder))

# VideoTestNet wraps TestNet with TemporalSmoother for EMA-based
# temporal consistency across consecutive frames
network = net.VideoTestNet(
    content_encoder, style_encoder, modulator, decoder,
    momentum=args.temporal_momentum)
network.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

# ------------------------------------------------------------------
# Process frames
# ------------------------------------------------------------------
# Pre-load and pre-process the first style image for timing
style_cache = None
prev_style_path = None

print(f"Temporal momentum: {args.temporal_momentum}")
print(f"Scene cuts at frames: {sorted(scene_cuts) if scene_cuts else 'none'}")
print("Processing...")

timings = []

for idx, (content_path, style_path) in enumerate(
        tqdm(zip(content_paths, style_paths), total=len(content_paths))):

    # --- Scene cut detection: reset smoother ---
    if idx in scene_cuts:
        network.reset_smoother()

    # --- Load and preprocess content frame ---
    content = content_tf(Image.open(str(content_path)).convert('RGB'))
    content = content.to(device).unsqueeze(0)

    # --- Load and preprocess style image (with caching) ---
    if style_path != prev_style_path:
        style = style_tf(Image.open(str(style_path)).convert('RGB'))
        style = style.to(device).unsqueeze(0)
        style_cache = style
        prev_style_path = style_path
    else:
        style = style_cache

    # --- Inference with temporal smoothing ---
    with torch.no_grad():
        torch.cuda.synchronize()
        tic = time.time()

        output = network(content, style, args.alpha)

        torch.cuda.synchronize()
        elapsed = time.time() - tic
        timings.append(elapsed)

    # --- Save output frame ---
    output = output.cpu()
    frame_name = f"{idx:06d}{args.save_ext}"
    save_image(output, str(output_dir / frame_name))

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
if timings:
    avg_time = sum(timings) / len(timings)
    print(f"\nProcessed {len(content_paths)} frames")
    print(f"Average time per frame: {avg_time:.4f}s")
    print(f"Total time: {sum(timings):.2f}s")
    print(f"Output saved to: {output_dir}/")
