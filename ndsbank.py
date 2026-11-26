"""ndsbank.py — an SBNK (+ its SWARs) as a SoundFont 2, for midi2taud.

One SF2 preset per VARIANT: an SBNK entry (the SSEQ's patch) together with the
ADSR overrides its track had in force (commands D0–D3 replace every region's
envelope for the notes they precede, SND_seq.c TrackPlayNote) — the overrides
become the variant's zone envelopes, so they reach midi2taud as ordinary presets.

Regions → zones. A drum set (record 16) is a zone per key; a key split (17) a
zone per range; anything else one zone over 0…127. Every zone has the region's
key range, root key (its note number: rootMidiKey), pan, envelope and loop.

Samples:
  * PCM8 / PCM16 / IMA-ADPCM SWAVs decoded to 16 bits (ndssdat); the rate is
    what the DS plays — ARM7_CLOCK/2 ÷ the SWAV's timer — not its nominal field.
  * PSG square (record 2): the duty table of SSEQPlayer's Channel.cpp
    wavedutytbl, the HIGH steps last, 8 steps a cycle; the driver's base timer
    is 8006 (SND_StartExChannelPsg), masked to 8004 on a PSG channel
    (SND_ExChannelMain), at the region's own key. Written 8 samples a step.
    Duty 7 is a constant (DC) on the DS: written as silence.
  * Noise (record 3): the 15-bit LFSR of SSEQPlayer's Channel.cpp (X=0x7FFF;
    X&1 → X>>1 ^ 0x6000, output −; else X>>1, output +), one step per timer
    period (8006) — its whole 32767-step period, each step held 4 samples, looped.

Envelope (SND_exChannel.c, one step per 5.2095 ms frame; levels in cB):
  * attack multiplies the level (−72.3 dB … 0) by coeff/256 a frame — slow at
    the top. SF2's attack is a linear amplitude ramp; it is given TWICE the
    DS's time to −6 dB, so both cross −6 dB together (and the ramp's end falls
    near the DS's −0.5 dB).
  * decay and release fall linearly in dB at rate/128 cB a frame: SF2's decay
    and release are linear in dB too, timed for 100 dB — 1000·128/rate frames
    (decay: rounded to the whole frames the driver takes to reach the sustain).
  * sustain is SNDi_DecibelSquareTable[s]: the same 40·log10 law as velocity.
"""

import math
import struct

from ndsplay import ARM7_CLOCK, FRAME_SEC, PSG_TIMER, DB, attack_coeff, decay_coeff

TC_MIN, TC_MAX = -12000, 8000
DECAY_FLOOR = 0.006        # s to the sustain level; see decay_tc

# Channel.cpp wavedutytbl (rows 0…7; −/+ 0x7FFF)
DUTY = [[-1] * (7 - d) + [1] * (d + 1) for d in range(7)] + [[-1] * 8]

PSG_STEP_RATE = ARM7_CLOCK / 2.0 / (PSG_TIMER & 0xFFFC)   # 2093.6 steps/s at the root key
NOISE_STEP_RATE = ARM7_CLOCK / 2.0 / PSG_TIMER
PSG_SPS = 8                                           # samples per duty step
NOISE_SPS = 4


def _tc(seconds):
    if seconds <= 0.0011:
        return TC_MIN
    return max(TC_MIN, min(TC_MAX, round(1200.0 * math.log2(seconds))))


def attack_tc(a):
    """SND_UpdateExChannelEnvelope's attack: att = -((-att * coeff) >> 8) a frame
    from -92544 (= -723 cB << 7)."""
    coeff = attack_coeff(a & 0x7F)
    att = -92544
    frames = 0
    while att and frames < 100000:
        att = -((-att * coeff) >> 8)
        frames += 1
        if (att >> 7) >= -60:           # -6 dB
            break
    if frames <= 1:
        return TC_MIN
    return _tc(2.0 * frames * FRAME_SEC)


def fall_tc(r):
    """Release: SF2's 100 dB time at the driver's rate/128 cB a frame."""
    rate = decay_coeff(r & 0x7F)
    return _tc(1000.0 * 128.0 / rate * FRAME_SEC)


def decay_tc(d, s):
    """Decay, timed so the sustain level is reached on the FRAME the driver
    reaches it (a whole number of frames, at least one — even the 'instant'
    rate 0xFFFF takes a frame), but never under DECAY_FLOOR: midi2taud splits a
    decay into at least three segments and rounds each to the envelope's
    3.9 ms steps, and a segment that rounds to zero is a TERMINATOR in a Taud
    volume envelope — the note would hold its peak and never fall to the
    sustain level. (The engine walks envelopes once a tick, so 6 ms is below
    anything it can draw anyway.)"""
    rate = decay_coeff(d & 0x7F)
    sus = sustain_cb(s)
    if sus <= 0:
        return fall_tc(d)
    frames = math.ceil(sus * 128.0 / rate)
    return _tc(max(frames * FRAME_SEC, DECAY_FLOOR) * 1000.0 / sus)


def sustain_cb(s):
    return max(0, min(1440, -DB[s & 0x7F]))


class _Sample:
    __slots__ = ('name', 'data', 'rate', 'corr', 'loop')


def _rate_of(exact):
    r = max(1, round(exact))
    corr = round(1200.0 * math.log2(exact / r))
    return r, max(-99, min(99, corr))


def _psg_sample(duty):
    s = _Sample()
    s.name = f'PSG duty {duty}'
    cyc = []
    for step in DUTY[duty]:
        cyc += [0 if duty == 7 else step * 0x7FFF] * PSG_SPS
    s.data = cyc * 4
    s.rate, s.corr = _rate_of(PSG_STEP_RATE * PSG_SPS)
    s.loop = (len(cyc), len(s.data))
    return s


def _noise_sample():
    s = _Sample()
    s.name = 'Noise LFSR'
    x = 0x7FFF
    out = []
    for _ in range(32767):
        if x & 1:
            x = (x >> 1) ^ 0x6000
            v = -0x7FFF
        else:
            x >>= 1
            v = 0x7FFF
        out += [v] * NOISE_SPS
    s.data = out
    s.rate, s.corr = _rate_of(NOISE_STEP_RATE * NOISE_SPS)
    s.loop = (0, len(out))
    return s


def _pcm_sample(w, name):
    s = _Sample()
    s.name = name
    s.data = w.pcm
    s.rate, s.corr = _rate_of(ARM7_CLOCK / 2.0 / max(1, w.timer))
    if w.loop and w.loop_len >= 2:
        a = min(w.loop_start, len(w.pcm))
        b = min(w.loop_start + w.loop_len, len(w.pcm))
        s.loop = (a, b) if b - a >= 2 else None
    else:
        s.loop = None
    return s


def _name20(s):
    return s.encode('ascii', 'replace')[:20].ljust(20, b'\0')


def build_sf2(path, variants, entries, swars, title='NDS bank', bank_name=''):
    """variants: list of ((patch, (a, d, s, r)), (bank, prog), key offset) —
    the offset moves the zones' key ranges and root keys together.
    Writes `path`; returns the number of presets written."""
    samples = []
    sample_index = {}

    def sample_for(region):
        if region.record == 1:
            sw = swars[region.swar] if region.swar < len(swars) else None
            w = sw.swav(region.swav) if sw is not None else None
            if w is None:
                return None
            key = ('pcm', region.swar, region.swav)
            if key not in sample_index:
                sample_index[key] = len(samples)
                samples.append(_pcm_sample(w, f'W{region.swar}:{region.swav}'))
        elif region.record == 2:
            key = ('psg', region.swav & 7)
            if key not in sample_index:
                sample_index[key] = len(samples)
                samples.append(_psg_sample(region.swav & 7))
        elif region.record == 3:
            key = ('noise',)
            if key not in sample_index:
                sample_index[key] = len(samples)
                samples.append(_noise_sample())
        else:
            return None
        return sample_index[key]

    insts = []            # (name, [zone gens])
    presets = []          # (name, bank, prog, inst index)
    for (patch, adsr), (bank, prog), koff in variants:
        if patch >= len(entries):
            continue
        e = entries[patch]
        zones = []
        for r in e.regions:
            si = sample_for(r)
            if si is None:
                continue
            a = r.a if adsr[0] == 0xFF else adsr[0]
            d = r.d if adsr[1] == 0xFF else adsr[1]
            s = r.s if adsr[2] == 0xFF else adsr[2]
            rr = r.r if adsr[3] == 0xFF else adsr[3]
            if rr == 0xFF:              # SND_NoteOn: release 0xFF = held (rate 0)
                rr = 0
            smp = samples[si]
            pan = max(-500, min(500, round((r.pan - 64) * 500 / 64)))
            lo, hi = min(127, r.lo + koff), min(127, r.hi + koff)
            gens = [(43, lo | (hi << 8)),
                    (17, pan),
                    (34, attack_tc(a)),
                    (36, decay_tc(d, s)),
                    (37, sustain_cb(s)),
                    (38, fall_tc(rr)),
                    (54, 1 if smp.loop else 0),
                    (58, min(127, r.note + koff)),
                    (53, si)]
            zones.append(gens)
        if not zones:
            continue
        kind = {16: 'drums', 17: 'split'}.get(e.record, {1: 'pcm', 2: 'psg', 3: 'noise'}.get(e.record, '?'))
        nm = f'P{patch:03d} {kind}'
        if adsr != (0xFF,) * 4:
            nm += ' ' + ''.join('-' if v == 0xFF else f'{v:02x}' for v in adsr)
        insts.append((nm, zones))
        presets.append((nm, bank, prog, len(insts) - 1))

    # ── sdta ──
    smpl = bytearray()
    shdr = bytearray()
    for s in samples:
        start = len(smpl) // 2
        smpl += struct.pack(f'<{len(s.data)}h', *[max(-32768, min(32767, v)) for v in s.data])
        smpl += bytes(46 * 2)
        end = start + len(s.data)
        ls, le = (start + s.loop[0], start + s.loop[1]) if s.loop else (start, end)
        shdr += _name20(s.name) + struct.pack('<IIIIIBbHH', start, end, ls, le, s.rate,
                                              60, s.corr, 0, 1)
    shdr += _name20('EOS') + bytes(26)

    # ── pdta ──
    phdr = bytearray(); pbag = bytearray(); pgen = bytearray()
    for i, (nm, bank, prog, ii) in enumerate(presets):
        phdr += _name20(nm) + struct.pack('<HHHIII', prog, bank, i, 0, 0, 0)
        pbag += struct.pack('<HH', len(pgen) // 4, 0)
        pgen += struct.pack('<HH', 41, ii)
    phdr += _name20('EOP') + struct.pack('<HHHIII', 0, 0, len(presets), 0, 0, 0)
    pbag += struct.pack('<HH', len(pgen) // 4, 0)
    pgen += bytes(4)
    inst = bytearray(); ibag = bytearray(); igen = bytearray()
    nbag = 0
    for nm, zones in insts:
        inst += _name20(nm) + struct.pack('<H', nbag)
        for gens in zones:
            ibag += struct.pack('<HH', len(igen) // 4, 0)
            nbag += 1
            for op, amt in gens:
                if op == 43:
                    igen += struct.pack('<HH', op, amt)
                elif op in (53, 54, 58):
                    igen += struct.pack('<HH', op, amt)
                else:
                    igen += struct.pack('<Hh', op, amt)
    inst += _name20('EOI') + struct.pack('<H', nbag)
    ibag += struct.pack('<HH', len(igen) // 4, 0)
    igen += bytes(4)
    pmod = bytes(10)
    imod = bytes(10)

    def chunk(cid, data):
        data = bytes(data)
        return cid + struct.pack('<I', len(data)) + data + (b'\0' if len(data) & 1 else b'')

    def lst(kind, parts):
        body = kind + b''.join(parts)
        return b'LIST' + struct.pack('<I', len(body)) + body

    def zstr(s):
        b = s.encode('latin-1', 'replace') + b'\0'
        return b + (b'\0' if len(b) & 1 else b'')

    info = lst(b'INFO', [chunk(b'ifil', struct.pack('<HH', 2, 1)),
                         chunk(b'isng', zstr('EMU8000')),
                         chunk(b'INAM', zstr(title[:250])),
                         chunk(b'ICMT', zstr(('2sf2taud: ' + bank_name)[:250]))])
    sdta = lst(b'sdta', [chunk(b'smpl', smpl)])
    pdta = lst(b'pdta', [chunk(b'phdr', phdr), chunk(b'pbag', pbag), chunk(b'pmod', pmod),
                         chunk(b'pgen', pgen), chunk(b'inst', inst), chunk(b'ibag', ibag),
                         chunk(b'imod', imod), chunk(b'igen', igen), chunk(b'shdr', shdr)])
    body = b'sfbk' + info + sdta + pdta
    with open(path, 'wb') as f:
        f.write(b'RIFF' + struct.pack('<I', len(body)) + body)
    return len(presets)
