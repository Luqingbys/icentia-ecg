import sys
import numpy as np
import pickle
import gzip
from itertools import chain
import random


def load_file(filename):
    # print(">", filename)
    data = pickle.load(gzip.open(filename))
    return data

def stream_array(data, chunk_size=5, shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    samples = (data.shape[0] // chunk_size) * chunk_size
    data = data[:samples]
    data = data.reshape(-1, chunk_size, data.shape[1])
    for i in range(data.shape[0]):
        yield data[i]


def buffered_random(stream, buffer_items=100, leak_percent=0.9):
    item_buffer = [None] * buffer_items
    leak_count = int(buffer_items * leak_percent)
    item_count = 0
    for item in stream:
        item_buffer[item_count] = item
        item_count += 1
        if buffer_items == item_count:
            random.shuffle(item_buffer)
            for item in item_buffer[leak_count:]:
                yield item
            item_count = leak_count
    if item_count > 0:
        item_buffer = item_buffer[:item_count]
        random.shuffle(item_buffer)
        for item in item_buffer:
            yield item


def stream_file_list(filenames, buffer_count=100, batch_size=10,
                     shuffle=True):
    if shuffle:
        random.shuffle(filenames)
    result = []
    streams = []
    while len(streams) < buffer_count and len(filenames) > 0:
        streams.append(stream_array(load_file(filenames.pop()),
                                    shuffle=shuffle))
    while len(streams) > 0 or len(filenames) > 0:
        i = 0
        while i < len(streams) and len(result) < batch_size:
            try:
                next_item = next(streams[i])
                i = (i + 1) % len(streams)
                result.append(next_item)
            except StopIteration:
                stream = None
                while len(filenames) > 0:
                    try:
                        stream = stream_array(load_file(filenames.pop()),
                                                  shuffle=shuffle)
                    except:
                        pass
                if stream is None:
                    streams = streams[:i] + streams[i+1:]
        if len(result) > 0:
            yield np.stack(result)
            result = []
        if shuffle:
            random.shuffle(streams)


if __name__ == "__main__":
    import glob
    from pprint import pprint
    directory = "/home/shawntan/projects/rpp-bengioy/jpcohen/icentia-ecg-dataset"
    filenames = glob.glob(directory + "/*_batched.npz")
    stream = stream_file_list(filenames)
    print(sum(x.shape[0] for x in stream))
