"""Render independent bounded audio grains on a compact programme clock."""
import numpy as np

WINDOW = 1024
HOP = 256
SEARCH = 192


def read(handle, start, count, total):
    left, right = max(0, start), min(total, start + count)
    result = np.zeros((count, 2), dtype=np.float64)
    if right > left:
        handle.seek(left * 8)
        raw = handle.read((right-left) * 8)
        if len(raw) != (right-left) * 8:
            raise ValueError('truncated dialogue stem')
        values = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite dialogue samples')
        result[left-start:right-start] = values
    return result



def channel_start(values):
    crossings = np.flatnonzero((values[:-1] <= 0) & (values[1:] > 0))
    crossings = crossings[crossings < 96]
    return int(crossings[0])+1 if len(crossings) else 0


def chunk(handle, start, count, speed, total, previous):
    left = int(start)
    needed = max(2, int(np.ceil(count*speed))+128)
    source = read(handle, left, needed, total)
    output = np.zeros((count, 2))
    for channel in range(2):
        offset = channel_start(source[:, channel])
        if speed >= 1:
            positions = np.arange(count)*speed+offset
            output[:, channel] = np.interp(positions, np.arange(needed), source[:, channel])
        else:
            grain = min(512, max(64, int(count*speed)))
            cursor = 0
            while cursor < count:
                size = min(grain, count-cursor)
                origin = int(cursor*speed)+offset
                available = source[origin:origin+size, channel]
                output[cursor:cursor+len(available), channel] = available
                cursor += size
        fade = min(64, count, len(previous))
        if fade:
            envelope = (np.arange(fade)+1)/(fade+1)
            output[:fade, channel] = (previous[-fade:, channel]*(1-envelope)
                                     + output[:fade, channel]*envelope)
    return output


def blocks(handle, occurrence, block_frames=4096):
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid dialogue block size')
    total = occurrence['source_samples']
    length = occurrence['output_samples']
    gain = occurrence['clip']['gain']
    points = occurrence['knots']
    if points == [[0, 0], [total, total]]:
        for start in range(0, length, block_frames):
            yield read(handle, start, min(block_frames, length-start), total)*gain
        return
    clock = occurrence.get('clock')
    ratio = occurrence.get('speed', total/length)
    previous = np.empty((0, 2))
    for origin in range(0, length, block_frames):
        count = min(block_frames, length-origin)
        if clock:
            row = clock[min(len(clock)-1, origin//1024)]
            start = row['source']+(origin-row['origin'])*row['speed']
        else:
            start = origin*ratio
        pcm = chunk(handle, start, count, ratio, total, previous)
        previous = pcm[-64:].copy()
        pcm *= gain
        if not np.isfinite(pcm).all():
            raise ValueError('nonfinite dialogue output')
        yield pcm.astype('<f4')
