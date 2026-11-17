"""Freeze source-to-picture dialogue timing independently of later edits.

Every recorder-marked source/picture knot is retained, not reduced to a single
average ratio, so mid-occurrence anchors reach their contracted picture
instants. A deterministic synthesis schedule is attached at creation time;
older version-one payloads (endpoints only) are still readable and rebuilt on
the fly, keeping previously saved databases usable.
"""
import hashlib
import json
from . import db, timeline, stretch_signal

RATE = 48000


def create(conn, project_id, maps):
    snapshot = timeline.snapshot(conn, project_id)
    clips = snapshot['clips']
    if not clips or len(clips) > 8 or not isinstance(maps, list) or len(maps) != len(clips):
        raise ValueError('one timing map per occurrence, at most eight occurrences')
    saved = []
    total = 0
    for clip, points in zip(clips, maps):
        source = (clip['end_ms'] - clip['start_ms']) * 48
        if not isinstance(points, list) or not 2 <= len(points) <= 16:
            raise ValueError('two to sixteen source and picture timing points required')
        knots = []
        for point in points:
            a, b = point['source_ms'], point['picture_ms']
            if type(a) is not int or type(b) is not int or min(a, b) < 0:
                raise ValueError('timing positions must be integer milliseconds')
            knots.append([a * 48, b * 48])
        if knots[0] != [0, 0] or knots[-1][0] != source:
            raise ValueError('timing map must cover the complete trimmed source')
        for (a, b), (c, d) in zip(knots, knots[1:]):
            if c <= a or d <= b:
                raise ValueError('timing positions must strictly increase')
            if c-a < 3840 or d-b < 3840 or not .5 <= (c-a)/(d-b) <= 2:
                raise ValueError('timing spans need 80 ms and speed between one half and two')
        schedule = stretch_signal.build_schedule(
            [(float(a), float(b)) for a, b in knots])
        output_samples = knots[-1][1]
        saved.append(dict(clip=clip, source_samples=source,
                          output_samples=output_samples, offset=total,
                          knots=knots, schedule=schedule))
        total += output_samples
    if total > RATE * 30:
        raise ValueError('dialogue programme exceeds thirty seconds')
    payload = dict(version=2, snapshot=snapshot, occurrences=saved, samples=total)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    with db.write(conn):
        row = conn.execute('INSERT INTO stretch_plans(payload,payload_sha256) VALUES (?,?)',
                           (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
    return get(conn, row.lastrowid)


def get(conn, identity):
    row = conn.execute('SELECT * FROM stretch_plans WHERE id=?', (identity,)).fetchone()
    if row is None or hashlib.sha256(row['payload'].encode()).hexdigest() != row['payload_sha256']:
        raise ValueError('dialogue timing plan missing or damaged')
    payload = json.loads(row['payload'])
    if payload.get('version') not in (1, 2):
        raise ValueError('unsupported dialogue timing plan')
    if payload['version'] == 1:
        # Legacy payloads only stored endpoints; rebuild their linear schedule.
        for occurrence in payload['occurrences']:
            occurrence.setdefault(
                'schedule', stretch_signal.build_schedule(
                    [(float(a), float(b)) for a, b in occurrence['knots']]))
    return dict(payload, id=identity)
