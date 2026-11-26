#!/usr/bin/env python3
"""2sf2taud.py — Nintendo DS music (.2sf / .mini2sf) to Taud. EXPERIMENTAL, out of tree.

Usage:
    python3 2sf2taud.py song.mini2sf [out.taud] [--seq N] [--seconds S]
                        [--no-vibrato] [-v] [-- midi2taud options…]
    python3 2sf2taud.py some_dir/ [out_dir/]          # every .mini2sf in it
    python3 2sf2taud.py song.mini2sf --list           # the SDAT's sequences

A 2SF is the game's own sound driver and its sound archive (SDAT) in a ROM image;
the mini patches the number of the sequence (SSEQ) the driver plays. Nothing is
emulated here: the SSEQ is run by a port of the DS's own sound driver as pret
decompiled it (ndsplay.py) for its notes, channel allocation and per-frame
pitch, level and pan; the SBNK it plays through becomes a SoundFont
(ndsbank.py); and midi2taud.py — the in-tree converter, unmodified on disk —
turns the two into a .taud. The event list is handed to midi2taud directly
rather than as a .mid, so a song may use all 16 DS tracks without one of them
landing on MIDI's drum channel; the .sf2 is kept beside the .taud.

What carries over: notes and their exact lengths, velocity (the DS's and SF2's
velocity laws are the same 40·log10), track volume/expression/master volume and
the sequence's own volume (as CC7), pan (CC10 + the region's pan), pitch bend,
portamento and pitch sweeps, ties (as bends on the held note), vibrato (as
bends; --no-vibrato drops it), tremolo (CC11, downwards half only), auto-pan,
ADSR including the per-track overrides (as preset variants), the driver's
channel stealing, loops (as the song's loop), tempo (on a grid of one Taud tick
per sequence tick, so it stays a whole Taud BPM). What does not: the envelope
SHAPES (SF2's are the nearest — see ndsbank.py), a tied note's velocity rise,
the DS's interpolation-free playback, leading silence (midi2taud starts at the
first note), and anything the game does to the sequence from outside
(variables it sets — they read as −1, as the driver initialises them —
sequence-arc sound effects).
"""

import argparse
import os
import struct
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ndssdat import load_2sf, find_sdat, SDAT, SEQ_PATCH_OFFSET   # noqa: E402
from ndsplay import simulate, FRAME_SEC, SEQ_PPQN                   # noqa: E402
from ndsbank import build_sf2                                       # noqa: E402

CONVERTERS = os.path.normpath(HERE)
SUB = 4                      # MIDI ticks per sequence tick
PPQ = SEQ_PPQN * SUB
MIDI_CHANNELS = [c for c in range(64) if c != 9]


def load_midi2taud():
    """midi2taud.py as a module, with its per-channel state array widened from
    16 to 64 — the only thing between it and more than 15 melodic channels."""
    path = os.path.join(CONVERTERS, 'midi2taud.py')
    src = open(path, encoding='utf-8').read()
    old = 'chs = [_ChState() for _ in range(16)]'
    if src.count(old) != 1:
        sys.exit('error: midi2taud.py changed: cannot widen its channel array')
    src = src.replace(old, 'chs = [_ChState() for _ in range(64)]')
    if CONVERTERS not in sys.path:
        sys.path.insert(0, CONVERTERS)
    mod = types.ModuleType('midi2taud')
    mod.__file__ = path
    sys.modules['midi2taud'] = mod
    exec(compile(src, path, 'exec'), mod.__dict__)
    return mod


def seq_id_of(rom, tags, args):
    if args.seq is not None:
        return args.seq
    return struct.unpack_from('<H', rom, SEQ_PATCH_OFFSET)[0]


def build_events(p, loop, title, warn=print):
    """Player log → (merged events for midi2taud, variants, channel names)."""
    # tracks that sound, in the order the sequence opened them
    used = []
    for tid in p.order:
        if any(n.track == tid for n in p.notes) and tid not in used:
            used.append(tid)
    ch_of = {tid: MIDI_CHANNELS[i] for i, tid in enumerate(used)}
    names = {ch_of[t]: f'Track {t}' for t in used}

    # variants (patch + ADSR overrides) → (bank, prog)
    keys = sorted({(n.patch, n.adsr) for n in p.notes},
                  key=lambda k: (k[1] != (0xFF,) * 4, k[0], k[1]))
    # MIDI key 0 is Taud note word $0000, a sentinel rather than a note: a
    # variant that plays key 0 (a drum kit's lowest drum) is shifted up a key,
    # its zones with it (ndsbank), so every pitch stays where it was
    lowest = {}
    highest = {}
    for n in p.notes:
        k = (n.patch, n.adsr)
        lowest[k] = min(lowest.get(k, 127), n.key)
        highest[k] = max(highest.get(k, 0), n.key)
    koff = {k: 1 if lowest[k] == 0 and highest[k] < 127 else 0 for k in keys}
    variants = [(k, (i // 128, i % 128), koff[k]) for i, k in enumerate(keys)]
    slot = {k: bp for k, bp, _ in variants}

    ev = []                                   # (midi tick, order, event)

    def put(pos, order, e):
        ev.append((max(0, round(pos * SUB)), order, e))

    put(0, 0, ('title', title))
    # one tempo per tick, the last set there: the driver's default 120 must not
    # stand beside the sequence's own tempo at tick 0, or midi2taud fits its grid
    # to 120 and every other tempo rounds to a whole Taud BPM on that grid
    tempo_at = {}
    for pos, tempo in p.tempo_log:
        if tempo > 0:
            tempo_at[pos] = tempo
    for pos, tempo in sorted(tempo_at.items()):
        put(pos, 0, ('tempo', 60.0 / (SEQ_PPQN * 240.0 / tempo * FRAME_SEC)))
    if loop:
        put(loop[0], 0, ('loopstart',))
        put(loop[1], 0, ('loopend',))

    # pitch: per channel bend range from the largest offset it needs
    for tid in used:
        ch = ch_of[tid]
        curve = p.curves.get(tid, [])
        mx = max([abs(c[1]) for c in curve] + [0]) / 64.0
        rng = max(2, min(127, int(mx) + 1))
        if mx > 127:
            warn(f'track {tid}: pitch offsets reach {mx:.0f} semitones; clipped')
        for num, val in ((101, 0), (100, 0), (6, rng), (38, 0)):
            put(0, 1, ('cc', ch, num, val))
        last = [None] * 4
        for pos, bend, level, pan, trem, pmod in curve:
            b14 = max(0, min(16383, 8192 + round(bend / 64.0 / rng * 8192)))
            c7 = max(0, min(127, round(127 * 10 ** (level / 400.0))))
            c10 = max(0, min(127, 64 + pan + pmod))
            c11 = max(0, min(127, round(127 * 10 ** (min(0, trem) / 400.0))))
            for i, (k, v) in enumerate((('bend', b14), (7, c7), (10, c10), (11, c11))):
                if last[i] != v:
                    last[i] = v
                    put(pos, 1, ('bend', ch, v) if k == 'bend' else ('cc', ch, k, v))

    # notes: a key struck again while still held is closed first
    cur_prog = {}
    active = {}
    notes = sorted(p.notes, key=lambda n: (n.start, n.serial))
    ends = {}
    for n in notes:
        ch = ch_of[n.track]
        k = (ch, n.key + koff[(n.patch, n.adsr)])
        if k in active and ends[id(active[k])] > n.start:
            ends[id(active[k])] = n.start
        active[k] = n
        ends[id(n)] = n.end if n.end is not None else n.start
    dropped = 0
    for n in notes:
        if n.key > 127:
            dropped += 1
            continue
        ch = ch_of[n.track]
        bank, prog = slot[(n.patch, n.adsr)]
        if cur_prog.get(ch) != (bank, prog):
            cur_prog[ch] = (bank, prog)
            put(n.start, 1, ('cc', ch, 0, bank))
            put(n.start, 1, ('prog', ch, prog))
        end = max(ends[id(n)], n.start + 1.0 / SUB)
        key = n.key + koff[(n.patch, n.adsr)]
        put(n.start, 2, ('on', ch, key, min(127, n.vel)))
        put(end, 0, ('off', ch, key))
    if dropped:
        warn(f'{dropped} note(s) above MIDI key 127 dropped')
    put(p.end_tick, 3, ('eot',))
    ev.sort(key=lambda e: (e[0], e[1]))
    merged = [(t, i, e) for i, (t, _o, e) in enumerate(ev)]
    return merged, variants, names


def pick_grid(p):
    """midi2taud's --rpb/--speed. Taud's tempo is a whole BPM, and a tick lasts
    2.5/BPM s; with rpb × speed = 48 fine ticks a beat — one per sequence tick —
    the Taud BPM is twice the DS tempo, a whole number to within the DS's own
    frame clock (0.016 %). midi2taud's own choice fits the onsets and lets the
    tempo round: 64 on a 32-a-beat grid is 85.33 → 85, 0.4 % slow. The rows are
    32nd notes (rpb 8, speed 6) unless more than 5 % of the notes fall between
    them (rpb 16, speed 3); a DS tempo over 524 cannot be doubled into the
    register (≤ 1048), and then the grid is 24 a beat (rpb 4, speed 6)."""
    if max((t for _, t in p.tempo_log), default=120) > 524:
        return 4, 6
    on = [n.start for n in p.notes]
    off6 = sum(1 for x in on if x % 6) / max(1, len(on))
    return (8, 6) if off6 <= 0.05 else (16, 3)


def convert(path, out, args, m2t=None):
    rom, tags = load_2sf(path)
    sd = SDAT(rom, find_sdat(rom))
    seq = seq_id_of(rom, tags, args)
    if seq not in sd.seqs:
        sys.exit(f'error: sequence {seq} is not in the SDAT')
    info = sd.seqs[seq]
    sname = sd.seq_names.get(seq, f'SSEQ {seq}')
    title = tags.get('title') or sname
    print(f'{os.path.basename(path)}: sequence {seq} ({sname}), bank {info.bank}, '
          f'volume {info.vol}, player {info.ply}')
    entries, swars = sd.bank(info.bank)
    warn = (lambda m: print('  warning: ' + m))
    p, loop = simulate(sd.sseq_data(seq), entries, swars, sd.channel_mask(seq),
                       info.vol, info.ppr, max_seconds=args.seconds, warn=warn,
                       vibrato=args.vibrato)
    print(f'  {len(p.notes)} notes on {len({n.track for n in p.notes})} tracks, '
          f'{p.end_tick} ticks' + (f', loop {loop[0]}–{loop[1]}' if loop else ', no loop')
          + (f', {p.dropped} note(s) found no free channel' if p.dropped else ''))
    if args.solo_track is not None:                 # debugging: one track alone
        p.notes = [n for n in p.notes if n.track == args.solo_track]
        p.curves = {k: v for k, v in p.curves.items() if k == args.solo_track}
    if not p.notes:
        print('  nothing to convert')
        return
    merged, variants, names = build_events(p, loop, title, warn)
    stem = os.path.splitext(out)[0]
    sf2 = stem + '.sf2'
    n = build_sf2(sf2, variants, entries, swars, title=title,
                  bank_name=sd.bank_names.get(info.bank, f'SBNK {info.bank}'))
    print(f'  wrote {sf2} ({n} presets)')
    if m2t is None:
        m2t = load_midi2taud()
    m2t.parse_midi = lambda _p: (('ppq', PPQ), merged, names)
    argv = sys.argv
    grid = []
    if '--rpb' not in args.rest and '--speed' not in args.rest:
        rpb, speed = pick_grid(p)
        grid = ['--rpb', str(rpb), '--speed', str(speed)]
    sys.argv = (['midi2taud', stem + '.mid', sf2, out] + grid
                + (['-v'] if args.verbose else []) + args.rest)
    try:
        m2t.main()
    finally:
        sys.argv = argv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--seq', type=int, help='play this SSEQ number instead of the mini\'s')
    ap.add_argument('--list', action='store_true', help='list the SDAT\'s sequences')
    ap.add_argument('--seconds', type=float, default=900.0,
                    help='give up on a song that neither loops nor ends by then')
    ap.add_argument('--no-vibrato', dest='vibrato', action='store_false')
    ap.add_argument('--solo-track', type=int, help=argparse.SUPPRESS)
    ap.add_argument('-v', '--verbose', action='store_true')
    argv = sys.argv[1:]
    rest = []
    if '--' in argv:                     # everything after -- goes to midi2taud
        i = argv.index('--')
        argv, rest = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    args.rest = rest

    if args.list:
        rom, tags = load_2sf(args.input)
        sd = SDAT(rom, find_sdat(rom))
        cur = seq_id_of(rom, tags, args)
        for i, s in sorted(sd.seqs.items()):
            print(f"{'*' if i == cur else ' '}{i:5d}  bank {s.bank:4d}  vol {s.vol:3d}  "
                  f"{sd.seq_names.get(i, '')}")
        return

    if os.path.isdir(args.input):
        out_dir = args.output or args.input
        os.makedirs(out_dir, exist_ok=True)
        m2t = load_midi2taud()
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(('.mini2sf', '.2sf')):
                out = os.path.join(out_dir, os.path.splitext(f)[0] + '.taud')
                try:
                    convert(os.path.join(args.input, f), out, args, m2t)
                except SystemExit as e:
                    print(f'  {f}: {e}')
        return
    out = args.output or os.path.splitext(args.input)[0] + '.taud'
    convert(args.input, out, args)


if __name__ == '__main__':
    main()
