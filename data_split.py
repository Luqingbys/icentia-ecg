#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os


chunk_length = 10**6
no_of_chunks = 50

if __name__ == "__main__":
    in_filename = sys.argv[1]
    filename = in_filename.split('.')[0]
    out_filename = filename + "_batched.npz"
    out_annotationfile = filename + "_batched_lbls.npz"

    if os.path.isfile(out_filename):
        print("File exists ", out_filename)
        sys.exit()
    data = np.load(in_filename)

    signal = (data['signal'].astype(np.float32) /
              data['norm_factor'].astype(np.float32))

    btype_idxs = [data['qrs'][data['qrs']['bType']==btype_id]['index']
                  for btype_id in range(5)]


    length = signal.shape[0]
    chunks = length // chunk_length
    signal = signal[:chunks * chunk_length]

    selected_chunks = np.random.choice(chunks, min(chunks, no_of_chunks), replace=False)
    selected_chunks.sort()

    batched_signal = signal.reshape(chunks, chunk_length)
    selected_array = batched_signal[selected_chunks]

    annotations = []
    for s_idx in selected_chunks:
        chunk_btype_idxs = []
        for btype in range(5):
            start = s_idx * chunk_length
            end = start + chunk_length
            points = btype_idxs[btype][(start <=  btype_idxs[btype]) & (btype_idxs[btype] < end)]
            chunk_btype_idxs.append(points)
        annotations.append(chunk_btype_idxs)
    np.savez(out_filename, selected_array)
    np.savez(out_annotationfile, annotations)
