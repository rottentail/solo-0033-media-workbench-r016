"""Pitch-preserving dialogue time stretching on an exact picture clock.

The processor is a bounded, streaming WSOLA (waveform similarity overlap-add)
engine built on the standard source-continuity formulation:

* grain centres walk a fixed synthesis hop ``Hs`` on the output clock while
  their nominal analysis centres advance by ``Ha = speed * Hs`` through the
  source. A tolerance search shifts each analysis centre so the head of the new
  grain continues the waveform the previous grain placed one synthesis hop
  earlier; this keeps consecutive grains phase continuous and therefore keeps
  the source pitch instead of imposing the timing speed on it;
* every saved source/picture anchor is a locked grain whose analysis centre is
  placed exactly on the contracted source position, so word/phrase positions
  reach their contracted picture instants and any search displacement
  accumulated inside a span resets at the next anchor;
* the displacement is computed once on the joint mono signal and applied
  identically to both channels, preserving inter-channel phase and the
  original left/right balance - channels are never processed separately;
* a grain whose analysis span contains a detected attack is locked, and the
  post-onset continuity target is taken from that locked grain, so a short
  onset is rendered once and stays crisp rather than being duplicated or
  smeared while joins are smoothed;
* the source file is read strictly forward through a small rolling ring and
  overlap-add output is flushed through a second small ring, so the full
  programme is never resident in memory.
"""
import bisect

import numpy as np

RATE = 48000
GRAIN = 2048              # analysis/synthesis grain length (~42.7 ms)
HS = 256                  # fixed synthesis hop (~5.3 ms); 87.5% overlap
SEARCH = 192              # maximum analysis displacement (samples, ~4 ms)
HALF = GRAIN // 2
CONTINUITY = GRAIN - HS   # head/tail length matched between grains
RING_CAP = GRAIN + 4 * SEARCH + HS + 512
ONSET_BLOCK = 128         # 2.67 ms energy blocks for attack detection
NOISE_FLOOR = 1e-5
ONSET_THRESHOLD = 4.0     # flux must exceed the quiet baseline by this factor
BYTES_PER_FRAME = 8       # stereo float32


def build_schedule(knots):
    """Return WSOLA grain frames honouring every source/picture anchor.

    A frame is ``(synthesis_centre, nominal_analysis_centre, locked)``.
    Synthesis centres walk the fixed output hop ``HS``. The nominal analysis
    centre advances at the local piecewise source/picture speed. Frames that
    coincide with a saved anchor are locked: their analysis centre is the exact
    contracted source position, which both places the anchor precisely and
    resets intra-span search displacement.
    """
    last_out = int(knots[-1][1])
    anchor = {int(out): float(src) for src, out in knots}
    frames = []
    span = 0
    src_a, out_a = float(knots[0][0]), float(knots[0][1])
    src_b, out_b = float(knots[1][0]), float(knots[1][1])
    speed = (src_b - src_a) / (out_b - out_a)
    frame = 0
    while True:
        syn = frame * HS
        if syn > last_out:
            break
        while syn > out_b and span < len(knots) - 2:
            span += 1
            src_a, out_a = float(knots[span][0]), float(knots[span][1])
            src_b, out_b = float(knots[span + 1][0]), float(knots[span + 1][1])
            speed = (src_b - src_a) / (out_b - out_a)
        if syn in anchor:
            nominal = anchor[syn]
            locked = True
        else:
            nominal = src_a + speed * (syn - out_a)
            locked = False
        frames.append((int(syn), float(nominal), locked))
        frame += 1
    # Guarantee a final locked grain centred exactly on the programme end.
    if frames[-1][0] != last_out:
        frames.append((last_out, float(knots[-1][0]), True))
    return frames


def is_identity(knots):
    return all(abs(src - out) < 1e-9 for src, out in knots)


def _hann(length):
    index = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * np.pi * index / (length - 1))


WINDOW = _hann(GRAIN)


class SourceStream:
    """Forward-only stereo reader with a small rolling float64 ring.

    ``ingested`` is the number of source frames pulled so far. Reads only move
    the underlying file forward; out-of-range windows are zero padded, which
    also covers short/no-audio sources padded by the decoder contract.
    """

    def __init__(self, handle, total):
        self.handle = handle
        self.total = total
        self.ingested = 0
        self.ring = np.zeros((0, 2), dtype=np.float64)
        self.previous_mono = None
        self.flux_floor = 0.0
        self.onsets = []                   # source frame positions of attacks
        self._partial = np.zeros((0, 2), dtype=np.float64)

    def _ingest(self, count):
        wanted = count * BYTES_PER_FRAME
        chunks = []
        remaining = wanted
        while remaining:
            raw = self.handle.read(min(remaining, 1 << 16))
            if not raw:
                break
            chunks.append(raw)
            remaining -= len(raw)
        if not chunks:
            return 0
        blob = b''.join(chunks)
        frames = len(blob) // BYTES_PER_FRAME
        pcm = np.frombuffer(blob[:frames * BYTES_PER_FRAME],
                            dtype='<f4').reshape(frames, 2).astype(np.float64)
        if not np.isfinite(pcm).all():
            raise ValueError('nonfinite dialogue source sample')
        self._note_onsets(pcm)
        if len(self.ring) + frames <= RING_CAP:
            self.ring = np.concatenate([self.ring, pcm])
        else:
            overflow = len(self.ring) + frames - RING_CAP
            self.ring = np.concatenate([self.ring[overflow:], pcm])
        self.ingested += frames
        return frames

    def _note_onsets(self, pcm):
        """Adaptive relative spectral-flux attack detection on the mono sum."""
        if self._partial.size:
            pcm = np.concatenate([self._partial, pcm])
        usable = (len(pcm) // ONSET_BLOCK) * ONSET_BLOCK
        self._partial = pcm[usable:].copy()
        mono = pcm[:usable].mean(axis=1)
        for offset in range(0, usable, ONSET_BLOCK):
            block = mono[offset:offset + ONSET_BLOCK]
            diff = np.diff(block, prepend=(self.previous_mono
                                           if self.previous_mono is not None
                                           else block[:1]))
            flux = float(np.mean(np.maximum(diff, 0.0)))
            if self.previous_mono is None:
                self.flux_floor = max(flux, NOISE_FLOOR)
            elif flux <= self.flux_floor:
                self.flux_floor = 0.95 * self.flux_floor + 0.05 * flux
            self.previous_mono = block[-1:].copy()
            if flux > ONSET_THRESHOLD * self.flux_floor + NOISE_FLOOR:
                position = self.ingested + offset
                if not self.onsets or position - self.onsets[-1] >= ONSET_BLOCK:
                    self.onsets.append(position)
                self.flux_floor = flux

    def _ensure(self, frame):
        target = min(self.total, frame + 1)
        while self.ingested < target:
            if self._ingest(min(4096, target - self.ingested)) == 0:
                break

    def window(self, centre, length=GRAIN):
        """Return ``length`` stereo samples centred on source ``centre``.

        Frame ``i`` represents source position ``floor(centre)-length//2+i``;
        positions outside the materialized stem are zero padded.
        """
        if length % 2:
            raise ValueError('grain windows must have even length')
        half = length // 2
        first = int(np.floor(centre)) - half
        last = first + length
        self._ensure(max(0, min(self.total, last - 1)))
        out = np.zeros((length, 2), dtype=np.float64)
        lo = max(0, -first)
        hi = length - max(0, last - self.total)
        if hi > lo:
            ring_base = self.ingested - len(self.ring)
            indices = np.arange(first + lo, first + hi) - ring_base
            if indices.size and (indices.min() < 0 or indices.max() >= len(self.ring)):
                return np.zeros((length, 2), dtype=np.float64)
            out[lo:hi] = self.ring[indices]
        return out

    def claim_onset(self, nominal, used):
        """Return True once for the grain whose centre is nearest an onset.

        ``used`` collects onset indices already owned, so an attack locks a
        single grain rather than every grain whose window happens to contain
        it; neighbouring grains remain free to smooth their joins.
        """
        if not self.onsets:
            return False
        index = bisect.bisect_left(self.onsets, nominal)
        candidates = [i for i in (index - 1, index)
                      if 0 <= i < len(self.onsets)]
        if not candidates:
            return False
        best = min(candidates, key=lambda i: abs(self.onsets[i] - nominal))
        if best in used or abs(self.onsets[best] - nominal) > HALF:
            return False
        used.add(best)
        return True


def _continuity_delta(stream, nominal, prev_analysis_centre):
    """Pick the analysis displacement that continues the prior waveform.

    The target is the source segment one synthesis hop after the previous
    grain's analysis start. The whole search region is read once and every
    candidate displacement is scored in a single matrix product, keeping the
    per-grain cost bounded. One displacement is chosen on the mono sum and
    returned for the caller to apply identically to both stereo channels.
    """
    prev_start = int(round(prev_analysis_centre)) - HALF
    target_start = prev_start + HS
    target = stream.window(target_start + CONTINUITY / 2, CONTINUITY).mean(axis=1)
    energy = float(np.dot(target, target))
    nominal_start = int(np.floor(nominal)) - HALF
    region_left = nominal_start - SEARCH
    region = stream.window(
        region_left + (CONTINUITY + 2 * SEARCH) / 2,
        CONTINUITY + 2 * SEARCH).mean(axis=1)
    candidates = np.lib.stride_tricks.sliding_window_view(region, CONTINUITY)
    scores = candidates @ target
    deltas = np.arange(-SEARCH, SEARCH + 1)
    valid = ((nominal_start + deltas >= 0)
             & (nominal_start + deltas + CONTINUITY <= stream.total))
    scores = np.where(valid, scores, -np.inf)
    best = int(np.argmax(scores))
    base_score = float(scores[SEARCH])
    if energy < 1e-8 or not np.isfinite(base_score) \
            or scores[best] <= base_score * 1.02:
        return 0
    return int(deltas[best])


def _identity_blocks(handle, total, gain, block_frames):
    """Exact 1:1 passthrough (still bounded, still one sequential read)."""
    remaining = total
    while remaining:
        count = min(block_frames, remaining)
        raw = handle.read(count * BYTES_PER_FRAME)
        frames = len(raw) // BYTES_PER_FRAME
        if frames == 0:
            pcm = np.zeros((count, 2), dtype=np.float64)
        else:
            pcm = np.frombuffer(raw, dtype='<f4').reshape(frames, 2)
            if frames < count:
                pcm = np.concatenate(
                    [pcm, np.zeros((count - frames, 2), dtype=np.float64)])
        pcm = pcm.astype(np.float64) * gain
        if not np.isfinite(pcm).all():
            raise ValueError('nonfinite dialogue output')
        remaining -= count
        yield pcm.astype('<f4')


class OutputRing:
    """Rolling weighted overlap-add accumulator with bounded residence.

    A grain is identified by the output position of its window centre; its
    samples occupy the half-open span ``[centre-HALF, centre+HALF)``. The
    unsquared Hann window is COLA at hop ``HS``, so a fixed window sum
    normalizes the overlap-add without amplitude modulation.
    """

    def __init__(self, length, gain, block_frames):
        self.length = length
        self.gain = gain
        self.block_frames = block_frames
        self.capacity = GRAIN + HS * 2
        self.sums = np.zeros((self.capacity, 2), dtype=np.float64)
        self.weights = np.zeros(self.capacity, dtype=np.float64)
        self.base = 0
        self.emitted = 0

    def add_grain(self, centre, grain):
        first = max(0, centre - HALF)
        last = min(self.length, centre + HALF)
        if last <= first:
            return
        g_lo = first - (centre - HALF)
        g_hi = g_lo + (last - first)
        if first < self.base:
            raise ValueError('dialogue schedule is not monotonic')
        indices = np.arange(first, last) - self.base
        if indices[-1] >= self.capacity:
            grow = indices[-1] + 1 - self.capacity
            self.sums = np.concatenate(
                [self.sums, np.zeros((grow, 2), dtype=np.float64)])
            self.weights = np.concatenate(
                [self.weights, np.zeros(grow, dtype=np.float64)])
            self.capacity += grow
        piece = grain[g_lo:g_hi] * WINDOW[g_lo:g_hi, None]
        self.sums[indices] += piece
        self.weights[indices] += WINDOW[g_lo:g_hi]

    def flush(self, upto):
        ready = max(0, min(upto, self.length) - self.emitted)
        if ready <= 0:
            return
        indices = np.arange(self.emitted, self.emitted + ready) - self.base
        acc = self.sums[indices]
        weight = self.weights[indices]
        pcm = np.zeros((ready, 2), dtype=np.float64)
        safe = weight > 1e-12
        pcm[safe] = acc[safe] / weight[safe, None]
        pcm *= self.gain
        if not np.isfinite(pcm).all():
            raise ValueError('nonfinite dialogue output')
        for start in range(0, ready, self.block_frames):
            yield pcm[start:start + self.block_frames].astype('<f4')
        self.emitted += ready
        drop = max(0, self.emitted - self.base - (HALF + HS))
        if drop:
            self.sums = np.roll(self.sums, -drop, axis=0)
            self.sums[-drop:] = 0
            self.weights = np.roll(self.weights, -drop)
            self.weights[-drop:] = 0
            self.base += drop


def synthesize(handle, occurrence, block_frames=4096):
    """Yield stereo float32 blocks for one frozen occurrence.

    Synthesis is deterministic and independent of ``block_frames`` (which only
    sizes I/O); the full occurrence is generated from its saved schedule while
    only grains and the two rolling rings stay resident.
    """
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid dialogue block size')
    total = occurrence['source_samples']
    length = occurrence['output_samples']
    gain = float(occurrence['clip']['gain'])
    knots = [(float(src), float(out)) for src, out in occurrence['knots']]

    if is_identity(knots):
        yield from _identity_blocks(handle, total, gain, block_frames)
        return

    schedule = occurrence.get('schedule')
    if not schedule:
        schedule = build_schedule(knots)

    stream = SourceStream(handle, total)
    output = OutputRing(length, gain, block_frames)
    prev_centre = None
    used_onsets = set()

    for syn, nominal, locked in schedule:
        # Read the nominal span first; ingestion also performs onset
        # detection, so the attack is known before this grain claims it.
        stream.window(nominal, GRAIN)
        onset_locked = stream.claim_onset(nominal, used_onsets)
        # Lock at anchors, at the first grain, or when this grain owns an
        # attack; otherwise continue the prior waveform.
        if locked or prev_centre is None or onset_locked:
            centre = float(nominal)
        else:
            centre = nominal + _continuity_delta(stream, nominal, prev_centre)
        grain = stream.window(centre, GRAIN)
        output.add_grain(syn, grain)
        prev_centre = centre
        # All positions at or below this centre are now covered.
        yield from output.flush(syn)

    yield from output.flush(length)
    if output.emitted != length:
        raise ValueError('incomplete dialogue synthesis')
