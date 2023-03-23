import sys
import numpy as np
import pickle
import gzip
from itertools import chain
import random
import threading
import queue

import multiprocessing


def multiprocess(stream, fun, queue_size=10, worker_count=5):
    in_queue = multiprocessing.JoinableQueue(maxsize=queue_size)
    out_queue = multiprocessing.JoinableQueue(maxsize=queue_size)
    end_marker = object()
    def producer():
        for item in stream:
            in_queue.put(item)
        for _ in range(worker_count):
            in_queue.put(end_marker)
    in_thread = multiprocessing.Process(target=producer)
    in_thread.daemon = True
    in_thread.start()

    def consumer():
        while True:
            item = in_queue.get()
            in_queue.task_done()
            if item is end_marker:
                out_queue.put(end_marker)
                break
            else:
                out_queue.put(fun(item))

    workers = [multiprocessing.Process(target=consumer)
               for _ in range(worker_count)]
    for w in workers:
        w.daemon = True
        w.start()

    end_count = 0
    while end_count < worker_count:
        item = out_queue.get()
        out_queue.task_done()
        if item is end_marker:
            end_count += 1
        else:
            yield item


def multithreaded(stream, fun, queue_size=10, worker_count=5):
    in_queue = queue.Queue(maxsize=queue_size)
    out_queue = queue.Queue(maxsize=queue_size)
    end_marker = object()
    def producer():
        for item in stream:
            in_queue.put(item)
        for _ in range(worker_count):
            in_queue.put(end_marker)
    in_thread = threading.Thread(target=producer)
    in_thread.daemon = True
    in_thread.start()

    def consumer():
        while True:
            item = in_queue.get()
            in_queue.task_done()
            if item is end_marker:
                out_queue.put(end_marker)
                break
            else:
                out_queue.put(fun(item))

    workers = [threading.Thread(target=consumer) for _ in range(worker_count)]
    for w in workers:
        w.daemon = True
        w.start()

    end_count = 0
    while end_count < worker_count:
        item = out_queue.get()
        out_queue.task_done()
        if item is end_marker:
            end_count += 1
        else:
            yield item


def threaded(stream, queue_size=10):
    '''线程，分线程读取数据'''
    work_queue = queue.Queue(maxsize=queue_size)
    end_marker = object()
    def producer():
        for item in stream:
            work_queue.put(item)
        work_queue.put(end_marker)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()
    # run as consumer
    item = work_queue.get()
    while item is not end_marker:
        yield item
        work_queue.task_done()
        item = work_queue.get()
    thread.join()

# 00009_batched.pkl.gz
# 00009_batched.pkl.gz

def load_file(filename):
    '''
    导入指定的文件
    返回一个ndarray类型的数组，是文件的读取结果，尺寸为(50, 1048577)
    '''
    # print(">", filename)
    data = pickle.load(gzip.open(filename))
    
    return data

def stream_array(data, chunk_size=5, shuffle=True):
    ''' 
    data: 数据，(50, 1048577)
    '''
    if shuffle:
        np.random.shuffle(data)
    samples = (data.shape[0] // chunk_size) * chunk_size # 默认为50
    data = data[:samples]
    data = data.reshape(-1, chunk_size, data.shape[1]) # (10, 5, 1048577)
    for i in range(data.shape[0]):
        yield data[i] # data[i]: (5, 1048577)


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


def stream_file_list(filenames, buffer_count=20, batch_size=10,
                     chunk_size=1,
                     shuffle=True):
    ''' 
    filenames: list，文件名列表，默认为'/icentia-ecg/datasets/'下的文件名列表
    '''
    filenames = filenames.copy()
    if shuffle:
        # 将文件名列表中的元素随机打乱
        random.shuffle(filenames)

    def _loaded_files():
        '''载入数据文件'''
        for i, fname in enumerate(filenames):
            # print("Loading", fname) fname: 00000_batched_lbls.pkl.gz
            yield i, load_file(fname)
    loaded_files = threaded(_loaded_files(), queue_size=20)

    result = []
    streams = []


    total_files = len(filenames)
    curr_file_idx = -1
    def make_stream():
        i, filedata = next(loaded_files)
        stream = stream_array(filedata,
                              shuffle=shuffle,
                              chunk_size=chunk_size)
        # print("Stream made.", i)
        return i, stream

    while len(streams) < buffer_count and curr_file_idx + 1 < total_files:
        try:
            curr_file_idx, stream = make_stream() # stream: (5, 1048577)
            streams.append(stream)
        except IOError:
            pass
        except EOFError:
            pass

    while len(streams) > 0 or curr_file_idx + 1 < total_files:
        i = 0
        while len(result) < batch_size and len(streams) > 0:
            try:
                i = i % len(streams)
                next_item = next(streams[i])
                i = (i + 1) % len(streams)
                result.append(next_item)
            except StopIteration:
                # print("Need to looad new file")
                stream = None
                while curr_file_idx + 1 < total_files:
                    try:
                        curr_file_idx, stream = make_stream()
                        break
                    except IOError:
                        pass
                    except EOFError:
                        pass
                if stream is not None:
                    streams[i] = stream
                else:
                    streams = streams[:i] + streams[i+1:]
        if len(result) > 0:
            if all(x.shape == result[0].shape for x in result):
                yield np.stack(result)
            result = []
        if shuffle:
            random.shuffle(streams)


if __name__ == "__main__":
    import glob
    from pprint import pprint
    directory = "/home/shawntan/projects/rpp-bengioy/jpcohen/icentia12k"
    print(directory + "/*_batched.pkl.gz")
    filenames = glob.glob(directory + "/*_batched.pkl.gz")
    print(filenames)
    stream = stream_file_list(filenames)
    print(sum(x.shape[0] for x in stream))
