"""ndssdat.py — the 2SF container and the Nitro Composer sound archive (SDAT).

A .mini2sf/.2sf is a PSF (version 0x24) whose program section is a sequence of
(u32 offset, u32 size, bytes) loads into a ROM image; a mini loads its _lib
first. The rips here (2sf.joshw.info sets) are all one layout: a sound driver
followed by the game's SDAT at ROM 0xE919C, and each mini a 2-byte patch at ROM
856000 (0xD0FC0) — the SSEQ number the driver plays. That number is checked
against the SDAT's own symbols (Mystery Dungeon names its files after them).

The SDAT layouts follow SSEQPlayer (tsvm reference_materials/nds-sseq/,
SDAT.cpp, INFOEntry.cpp, SBNK.cpp, SWAR.cpp, SWAV.cpp, SSEQ.cpp).
"""

import os
import struct
import zlib

SEQ_PATCH_OFFSET = 856000          # where every rip here writes the SSEQ id


# ── PSF / 2SF ────────────────────────────────────────────────────────────────

def read_psf(path):
    d = open(path, 'rb').read()
    if d[:3] != b'PSF':
        raise ValueError(f'{path}: not a PSF file')
    if d[3] != 0x24:
        raise ValueError(f'{path}: PSF version {d[3]:#x}, not 2SF (0x24)')
    rsz, psz, _crc = struct.unpack_from('<III', d, 4)
    prog = d[16 + rsz:16 + rsz + psz]
    tags = {}
    t = d[16 + rsz + psz:]
    if t[:5] == b'[TAG]':
        for line in t[5:].decode('utf-8', 'replace').split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip().lower()
                v = v.strip()
                tags[k] = (tags[k] + '\n' + v) if k in tags else v
    return (zlib.decompress(prog) if prog else b''), tags


def _find_case_insensitive(directory, name):
    p = os.path.join(directory, name)
    if os.path.exists(p):
        return p
    low = name.lower()
    for f in os.listdir(directory):
        if f.lower() == low:
            return os.path.join(directory, f)
    raise FileNotFoundError(p)


def _apply_loads(rom: bytearray, prog: bytes):
    pos = 0
    while pos + 8 <= len(prog):
        off, size = struct.unpack_from('<II', prog, pos)
        pos += 8
        data = prog[pos:pos + size]
        pos += size
        if off + len(data) > len(rom):
            rom.extend(bytes(off + len(data) - len(rom)))
        rom[off:off + len(data)] = data


def load_2sf(path):
    """→ (rom bytearray, tags of THIS file). _lib first, then _lib2…, then self
    (the PSF library order), each load written over the one before."""
    rom = bytearray()
    tags = _load_into(path, rom, 0)
    return rom, tags


def _load_into(path, rom, _depth):
    if _depth > 10:
        raise ValueError('2SF _lib chain too deep')
    prog, tags = read_psf(path)
    d = os.path.dirname(os.path.abspath(path))
    libs = []
    if '_lib' in tags:
        libs.append(tags['_lib'])
    n = 2
    while f'_lib{n}' in tags:
        libs.append(tags[f'_lib{n}'])
        n += 1
    for lib in libs:
        _load_into(_find_case_insensitive(d, lib), rom, _depth + 1)
    _apply_loads(rom, prog)
    return tags


def find_sdat(rom) -> int:
    i = rom.find(b'SDAT\xff\xfe\x00\x01')
    if i < 0:
        raise ValueError('no SDAT in the ROM image')
    return i


# ── SDAT ─────────────────────────────────────────────────────────────────────

class SWAV:
    __slots__ = ('wave_type', 'loop', 'rate', 'timer', 'loop_start', 'loop_len', 'pcm')

    def samples(self):
        """Decoded int16 samples (list), loop_start/loop_len in samples —
        SSEQPlayer SWAV.cpp: PCM8 ×4 per word, PCM16 ×2, ADPCM ×8 after the
        4-byte header word (which the loop offset counts)."""
        return self.pcm


_IMA_INDEX = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)
_IMA_STEP = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
    3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767)


def decode_adpcm(raw: bytes):
    """SSEQPlayer SWAV.cpp DecodeADPCM: header = s16 predictor, u16 step index."""
    pred = struct.unpack_from('<h', raw, 0)[0]
    idx = struct.unpack_from('<H', raw, 2)[0]
    if idx > 88:
        idx = 88
    out = [0] * ((len(raw) - 4) * 2)
    o = 0
    for b in raw[4:]:
        for nib in (b & 0x0F, b >> 4):
            step = _IMA_STEP[idx]
            idx += _IMA_INDEX[nib]
            if idx < 0:
                idx = 0
            elif idx > 88:
                idx = 88
            diff = step >> 3
            if nib & 4:
                diff += step
            if nib & 2:
                diff += step >> 1
            if nib & 1:
                diff += step >> 2
            if nib & 8:
                pred -= diff
                if pred < -0x8000:
                    pred = -0x8000
            else:
                pred += diff
                if pred > 0x7FFF:
                    pred = 0x7FFF
            out[o] = pred
            o += 1
    return out


def parse_swav(d: bytes, p: int) -> SWAV:
    w = SWAV()
    w.wave_type, w.loop, w.rate, w.timer, loop_off, loop_len = struct.unpack_from('<BBHHHI', d, p)
    size = (loop_off + loop_len) * 4
    raw = d[p + 12:p + 12 + size]
    if w.wave_type == 0:
        w.pcm = [(b - 256 if b >= 128 else b) << 8 for b in raw]
        w.loop_start, w.loop_len = loop_off * 4, loop_len * 4
    elif w.wave_type == 1:
        w.pcm = list(struct.unpack_from(f'<{len(raw) // 2}h', raw, 0))
        w.loop_start, w.loop_len = loop_off * 2, loop_len * 2
    else:
        w.pcm = decode_adpcm(raw)
        w.loop_start = (loop_off - 1 if loop_off else 0) * 8
        w.loop_len = loop_len * 8
    return w


class Region:
    """One SBNK note definition (SBNKInstrument): a key range and its sound."""
    __slots__ = ('lo', 'hi', 'record', 'swav', 'swar', 'note', 'a', 'd', 's', 'r', 'pan')

    def __init__(self, lo, hi, record, d, p):
        self.lo, self.hi, self.record = lo, hi, record
        (self.swav, self.swar, self.note, self.a, self.d, self.s, self.r,
         self.pan) = struct.unpack_from('<HHBBBBBB', d, p)


class BankEntry:
    __slots__ = ('record', 'regions')


def parse_sbnk(d: bytes, base: int):
    if d[base:base + 4] != b'SBNK':
        raise ValueError('bad SBNK')
    count = struct.unpack_from('<I', d, base + 0x38)[0]
    entries = []
    for i in range(count):
        rec, off = struct.unpack_from('<BH', d, base + 0x3C + 4 * i)
        e = BankEntry()
        e.record = rec
        e.regions = []
        p = base + off
        if rec == 16:
            lo, hi = d[p], d[p + 1]
            p += 2
            for k in range(hi - lo + 1):
                r = struct.unpack_from('<H', d, p)[0]
                e.regions.append(Region(lo + k, lo + k, r, d, p + 2))
                p += 12
        elif rec == 17:
            ranges = d[p:p + 8]
            p += 8
            i2 = 0
            while i2 < 8 and ranges[i2]:
                r = struct.unpack_from('<H', d, p)[0]
                lo = ranges[i2 - 1] + 1 if i2 else 0
                e.regions.append(Region(lo, ranges[i2], r, d, p + 2))
                p += 12
                i2 += 1
        elif rec:
            e.regions.append(Region(0, 127, rec, d, p))
        entries.append(e)
    return entries


class SWAR:
    def __init__(self, d: bytes, base: int):
        if d[base:base + 4] != b'SWAR':
            raise ValueError('bad SWAR')
        self.d, self.base = d, base
        count = struct.unpack_from('<I', d, base + 0x38)[0]
        self.offsets = struct.unpack_from(f'<{count}I', d, base + 0x3C)
        self._cache = {}

    def __len__(self):
        return len(self.offsets)

    def swav(self, i):
        if i >= len(self.offsets) or not self.offsets[i]:
            return None
        if i not in self._cache:
            self._cache[i] = parse_swav(self.d, self.base + self.offsets[i])
        return self._cache[i]


class SeqInfo:
    __slots__ = ('file_id', 'bank', 'vol', 'cpr', 'ppr', 'ply')


class SDAT:
    def __init__(self, d: bytes, base: int = 0):
        if d[base:base + 4] != b'SDAT':
            raise ValueError('bad SDAT')
        self.d, self.base = d, base
        (symb_off, _, info_off, _, fat_off, _) = struct.unpack_from('<6I', d, base + 0x10)
        # FAT
        fp = base + fat_off
        nf = struct.unpack_from('<I', d, fp + 8)[0]
        self.fat = [struct.unpack_from('<II', d, fp + 12 + 16 * i) for i in range(nf)]
        # INFO
        ip = base + info_off
        rec = struct.unpack_from('<8I', d, ip + 8)

        def record(n):
            if not rec[n]:
                return {}
            rp = ip + rec[n]
            cnt = struct.unpack_from('<I', d, rp)[0]
            out = {}
            for i in range(cnt):
                eo = struct.unpack_from('<I', d, rp + 4 + 4 * i)[0]
                if eo:
                    out[i] = ip + eo
            return out
        self.seqs = {}
        for i, p in record(0).items():
            s = SeqInfo()
            s.file_id = struct.unpack_from('<H', d, p)[0]
            s.bank, s.vol, s.cpr, s.ppr, s.ply = struct.unpack_from('<HBBBB', d, p + 4)
            if not s.vol:
                s.vol = 0x7F            # INFOEntry.cpp: "Prevents nothing for volume"
            self.seqs[i] = s
        self.banks = {}
        for i, p in record(2).items():
            fid = struct.unpack_from('<H', d, p)[0]
            self.banks[i] = (fid, struct.unpack_from('<4H', d, p + 4))
        self.wavearcs = {i: struct.unpack_from('<H', d, p)[0] for i, p in record(3).items()}
        self.players = {}
        for i, p in record(4).items():
            self.players[i] = struct.unpack_from('<H', d, p + 2)[0]
        # SYMB (names; optional)
        self.seq_names, self.bank_names, self.wavearc_names = {}, {}, {}
        if symb_off:
            sp = base + symb_off
            srec = struct.unpack_from('<8I', d, sp + 8)

            def names(n):
                if not srec[n]:
                    return {}
                rp = sp + srec[n]
                cnt = struct.unpack_from('<I', d, rp)[0]
                out = {}
                for i in range(cnt):
                    so = struct.unpack_from('<I', d, rp + 4 + 4 * i)[0]
                    if so:
                        e = d.index(b'\0', sp + so)
                        out[i] = d[sp + so:e].decode('latin-1')
                return out
            self.seq_names = names(0)
            self.bank_names = names(2)
            self.wavearc_names = names(3)

    def file(self, fid):
        off, size = self.fat[fid]
        return self.base + off

    def sseq_data(self, seq_id):
        """→ bytes of the SSEQ's command stream (offsets inside it are relative
        to its start, as Goto/Call/OpenTrack address it)."""
        p = self.file(self.seqs[seq_id].file_id)
        if self.d[p:p + 4] != b'SSEQ':
            raise ValueError('bad SSEQ')
        size, data_off = struct.unpack_from('<II', self.d, p + 0x14)
        return self.d[p + data_off:p + 0x10 + size]

    def bank(self, bank_id):
        fid, wa = self.banks[bank_id]
        entries = parse_sbnk(self.d, self.file(fid))
        swars = []
        for w in wa:
            if w == 0xFFFF or w not in self.wavearcs:
                swars.append(None)
            else:
                swars.append(SWAR(self.d, self.file(self.wavearcs[w])))
        return entries, swars

    def channel_mask(self, seq_id):
        m = self.players.get(self.seqs[seq_id].ply, 0)
        return m or 0xFFFF
