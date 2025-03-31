import argparse
import os
from decord import VideoReader, cpu
import imageio
import cv2
from tqdm import tqdm
import numpy as np
import sys
def main(src, tgt):
    video = VideoReader(src, ctx=cpu(0))
    frames = video.get_batch(list(range(len(video)))).asnumpy()
    out_frames = []
    for f in tqdm(frames):
        f = cv2.resize(f, (1024, 576))
        out_frames.append(f)
    
    imageio.mimsave(
        tgt,
        np.stack(out_frames),
        fps=video.get_avg_fps(),
        # macro_block_size=1,
    )


if __name__ == "__main__": 
    datadir =  'video/showcase'
    files = os.listdir(datadir)
    files = filter(lambda f: f.startswith('film17') and f.endswith('output.mkv'), files)
    for f in files:
        main(os.path.join(datadir, f), os.path.join(datadir, f.replace('.mkv', '.mp4')))