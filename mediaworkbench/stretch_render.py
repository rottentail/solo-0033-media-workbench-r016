"""Render frozen dialogue timing with one decode and deterministic auditions.

Every render or audition first synthesizes the complete frozen programme
history into a private temporary file, then crops the requested window from
it. An audition therefore reproduces the complete delivery at the same
positions, regardless of block size, and later timeline edits cannot change a
frozen plan. Each trimmed occurrence is materialized and read exactly once,
strictly forward; only local windows and output blocks stay resident.
"""
import json
import os
from pathlib import Path
import tempfile
from . import stretch_plan, stretch_signal, media_sources, audio_signal
from .assets import fingerprint
from .tools import run


def render(conn, identity, destination, at_sample=0, samples=None, block_frames=4096):
    plan = stretch_plan.get(conn, identity)
    samples = plan['samples']-at_sample if samples is None else samples
    if (type(at_sample) is not int or type(samples) is not int or at_sample < 0
            or samples < 1 or at_sample+samples > plan['samples']):
        raise ValueError('dialogue audition outside programme')
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid dialogue block size')
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve()/target.name
    if os.path.lexists(target) or not target.parent.is_dir():
        raise ValueError('dialogue destination exists or directory missing')
    with tempfile.TemporaryDirectory(prefix='.stretch-', dir=target.parent) as directory:
        root = Path(directory)
        programme = root/'programme.f32'
        excerpt = root/'excerpt.f32'
        output = root/'dialogue.wav'
        selected = []
        # Full-history synthesis: one forward pass per occurrence, in order.
        with programme.open('wb') as handle:
            for occurrence in plan['occurrences']:
                clip = media_sources.select(conn, occurrence['clip'])
                clip['has_audio'] = any(s.get('codec_type') == 'audio'
                                        for s in json.loads(clip['metadata'])['streams'])
                selected.append(clip)
                with audio_signal.materialized(clip) as source, source.open('rb') as stem:
                    local = dict(occurrence, clip=clip)
                    for pcm in stretch_signal.synthesize(stem, local, block_frames):
                        handle.write(pcm.astype('<f4').tobytes())
        if programme.stat().st_size != plan['samples']*8:
            raise ValueError('incomplete dialogue output')
        # Crop the requested audition window from the deterministic history.
        with programme.open('rb') as history, excerpt.open('wb') as cropped:
            history.seek(at_sample*8)
            remaining = samples*8
            while remaining:
                chunk = history.read(min(remaining, 1 << 16))
                if not chunk:
                    raise ValueError('incomplete dialogue audition')
                cropped.write(chunk)
                remaining -= len(chunk)
        run('ffmpeg', ['-v', 'error', '-y', '-f', 'f32le', '-ar', '48000', '-ac', '2',
                       '-i', str(excerpt), '-c:a', 'pcm_f32le', str(output)])
        # Re-verify immutable source identity before publication.
        for clip in selected:
            media_sources.select(conn, clip)
        checksum = fingerprint(output)
        with output.open('rb') as handle:
            os.fsync(handle.fileno())
        os.link(output, target)
    return dict(plan_id=identity, path=str(target), samples=samples,
                at_sample=at_sample, rate=48000, channels=2, sha256=checksum)
