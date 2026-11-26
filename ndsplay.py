"""ndsplay.py — the Nitro sound driver's sequencer, run for its EVENTS rather than its sound.

A port of the DS's own ARM7 sound driver as decompiled by pret (pokediamond
arm7/lib/src/SND_seq.c, SND_exChannel.c, SND_bank.c, SND_util.c, SND_main.c;
copies in tsvm reference_materials/nds-sseq/) minus the hardware: every frame
(5.2095 ms: OS alarm period 2728 ticks of 64 cycles) does what SndThread does —
SND_SeqMain (PlayerSeqMain: the tempo counter and TrackStepTicks; then
PlayerUpdateChannel → TrackUpdateChannel), SND_ExChannelMain (envelope, sweep,
LFO, the release cut-off), SND_CalcRandom — and logs what a MIDI file carries:

  * a Note per channel a track sounds (track, key, velocity, the program and
    the track's envelope overrides in force), from the note-on to the moment the
    channel stops being held: the release (length run out, track end, tie off),
    the channel stolen by another note, or a one-shot sample running out;
  * per track, once a frame: its newest held channel's pitch offset (bend,
    sweep/portamento, a tied note's new key, a pitch LFO — 1/64 semitone,
    relative to the key that channel's note was logged at), the track's level
    (volume + expression + player volume + the sequence's own volume, cB), its
    pan, the channel's volume/pan LFO, and a tied note's velocity change.

Positions are SEQUENCE TICKS (48 per quarter) as floats: a note sits on the
integer tick that started it; frame events at the tempo counter's fraction.
Channel allocation (16 channels, priorities, stealing) is the driver's.

Where SSEQPlayer (in_xsf — the reference this began from) disagrees with the
decompiled driver, the driver wins: one attack step a frame, a quarter sine of
0x20, PSG/noise timer 8006, keys clamped not wrapped, a sweep without the
portamento flag, a track's notes released when it ends, release cut-off on the
channel's whole level.
"""

from fss_tables import PITCH_TBL, VOL_TBL, ATTACK_LUT, SCALE_LUT, SUST_LUT, SINE_LUT

ARM7_CLOCK = 33513982
FRAME_SEC = 64.0 * 2728.0 / ARM7_CLOCK
TIMER_RATE = 240
SEQ_PPQN = 48
PSG_TIMER = 8006                        # SND_StartExChannelPsg / Noise
ALLOC_ORDER = (4, 5, 6, 7, 2, 0, 3, 1, 8, 9, 10, 11, 14, 12, 15, 13)   # sChannelAllocationOrder
SHIFT = (0, 1, 2, 4)                    # sSampleDataShiftTable
DB = SUST_LUT                           # SNDi_DecibelSquareTable

ENV_ATTACK, ENV_DECAY, ENV_SUSTAIN, ENV_RELEASE = range(4)
LFO_PITCH, LFO_VOLUME, LFO_PAN = range(3)
INST_PCM, INST_PSG, INST_NOISE, INST_DIRECTPCM, INST_DUMMY = 1, 2, 3, 4, 5


def tdiv(a, b):
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def s8(x):
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def attack_coeff(attack):                # SND_SetExChannelAttack
    return (255 - attack) & 0xFF if attack < 109 else ATTACK_LUT[127 - attack]


def decay_coeff(v):                      # CalcDecayCoeff
    if v == 127:
        return 0xFFFF
    if v == 126:
        return 0x3C00
    if v < 50:
        return (v * 2 + 1) & 0xFFFF
    return (0x1E00 // (126 - v)) & 0xFFFF


def cnv_scale(v):                        # the NNS library's linear dB table (SSEQPlayer)
    if v & 0x80:
        v = 0x7F
    return SCALE_LUT[v]


def sin_idx(x):                          # SND_SinIdx
    if x < 0x20:
        return SINE_LUT[x]
    if x < 0x40:
        return SINE_LUT[0x40 - x]
    if x < 0x60:
        return -SINE_LUT[x - 0x40]
    return -SINE_LUT[0x20 - (x - 0x60)]


def calc_timer(timer, pitch):            # SND_CalcTimer
    octave = 0
    p = -pitch
    while p < 0:
        octave -= 1
        p += 768
    while p >= 768:
        octave += 1
        p -= 768
    r = (PITCH_TBL[p] + 0x10000) * timer
    shift = octave - 16
    if shift <= 0:
        r >>= -shift
    elif shift < 32:
        if r & ((~0 << (32 - shift)) & 0xFFFFFFFFFFFFFFFF):
            return 0xFFFF
        r <<= shift
    else:
        return 0xFFFF
    return min(max(r, 0x10), 0xFFFF)


def calc_channel_volume(v):              # SND_CalcChannelVolume → (vol | div << 8)
    v = min(max(v, -723), 0)
    r = VOL_TBL[v + 723]
    div = 3 if v < -240 else 2 if v < -120 else 1 if v < -60 else 0
    return r | (div << 8)


class Note:
    __slots__ = ('track', 'key', 'vel', 'patch', 'adsr', 'start', 'end', 'why', 'serial')

    def __repr__(self):
        return (f'Note(t{self.track} key{self.key} v{self.vel} p{self.patch} '
                f'{self.start:.2f}–{self.end} {self.why})')


class Lfo:
    __slots__ = ('target', 'depth', 'range', 'speed', 'delay', 'counter', 'delay_counter')

    def __init__(self):
        self.target, self.depth, self.range, self.speed, self.delay = LFO_PITCH, 0, 1, 16, 0
        self.counter = self.delay_counter = 0

    def param_from(self, m):
        self.target, self.depth, self.range, self.speed, self.delay = \
            m.target, m.depth, m.range, m.speed, m.delay


class ExChannel:
    def __init__(self, i):
        self.id = i
        self.active = False
        self.start = False
        self.hw_active = False
        self.priority = 0
        self.volume = 0
        self.track = None                 # owning track (callback user data) or None
        self.note = None
        self.serial = 0
        self.setup(0)

    def setup(self, priority):            # ExChannelSetup
        self.length = 0
        self.priority = priority
        self.volume = 127
        self.start = False
        self.auto_sweep = True
        self.midi_key = 60
        self.root_key = 60
        self.velocity = 127
        self.init_pan = 0
        self.user_decay = self.user_decay2 = self.user_pitch = self.user_pan = 0
        self.pan_range = 127
        self.sweep_pitch = self.sweep_length = self.sweep_counter = 0
        self.env_attack = attack_coeff(127)
        self.env_decay = decay_coeff(127)
        self.env_sustain = 127
        self.env_release = decay_coeff(127)
        self.lfo = Lfo()
        self.env_att = -92544
        self.env_status = ENV_ATTACK
        self._sync_start = False
        self.kind = 0
        self.timer = 0
        self.wave_timer = 0
        self.wave_len = 0                 # PCM samples (one-shot end), 0 = endless
        self.play_pos = 0.0
        # logged per frame
        self.adj = 0
        self.lfo_vol = 0
        self.lfo_pan = 0


class Track:
    def __init__(self, player, idx):
        self.p = player
        self.idx = idx
        self.init()

    def init(self):                       # TrackInit
        self.base = None
        self.cur = None
        self.note_wait = True
        self.muted = False
        self.tie = False
        self.note_finish_wait = False
        self.portamento = False
        self.cmp = True
        self.stack = []                   # [pos, loopcount]
        self.program = 0
        self.priority = 64
        self.volume = 127
        self.expression = 127
        self.pan = 0
        self.pitch_bend = 0
        self.env = [0xFF, 0xFF, 0xFF, 0xFF]
        self.pan_range = 127
        self.bend_range = 2
        self.porta_key = 60
        self.porta_time = 0
        self.sweep_pitch = 0
        self.transpose = 0
        self.channel_mask = 0xFFFF
        self.mod = Lfo()
        self.wait = 0
        self.chans = []                   # channelLLHead list, head first
        self.visited = {}                 # command address → first tick
        self.loops = []                   # (target's first tick, jump tick, silent)
        self.note_ticks = []              # every tick a note was struck
        self.note_tick = None
        self.stopped = False
        self.stop_tick = None

    def r8(self):
        b = self.p.data[self.cur] if self.cur < len(self.p.data) else 0xFF
        self.cur += 1
        return b

    def r16(self):
        return self.r8() | (self.r8() << 8)

    def r24(self):
        return self.r8() | (self.r8() << 8) | (self.r8() << 16)

    def rvlv(self):
        r = 0
        while True:
            b = self.r8()
            r = (r << 7) | (b & 0x7F)
            if not (b & 0x80):
                return r

    def value(self, kind):                # TrackParseValue
        if kind == 'u8':
            return self.r8()
        if kind == 'u16':
            return self.r16()
        if kind == 'vlv':
            return self.rvlv()
        if kind == 'var':
            return self.p.var_get(self.r8())
        lo = s16(self.r16())
        hi = s16(self.r16())
        ran = self.p.rand()
        return ((ran * (hi - lo + 1)) >> 16) + lo

    # --- TrackPlayNote ---------------------------------------------------------
    def play_note(self, key, vel, length):
        p = self.p
        chn = None
        if self.tie:
            chn = self.chans[0] if self.chans else None
            if chn is not None:
                chn.midi_key = key
                chn.velocity = vel
        new = chn is None
        if new:
            inst = p.read_inst(self.program, key)
            if inst is None:
                return
            itype, region = inst
            if itype in (INST_PCM, INST_DIRECTPCM):
                allowed = 0xFFFF
            elif itype == INST_PSG:
                allowed = 0x3F00
            elif itype == INST_NOISE:
                allowed = 0xC000
            else:
                return
            allowed &= self.channel_mask
            chn = p.alloc_channel(allowed, p.prio + self.priority, self)
            if chn is None:
                p.dropped += 1
                return
            if not p.note_on(chn, key, vel, -1 if self.tie else length, itype, region):
                chn.priority = 0
                chn.track = None
                return
            self.chans.insert(0, chn)
            chn.track = self
        e = self.env
        if e[0] != 0xFF:
            chn.env_attack = attack_coeff(e[0])
        if e[1] != 0xFF:
            chn.env_decay = decay_coeff(e[1])
        if e[2] != 0xFF:
            chn.env_sustain = e[2]
        if e[3] != 0xFF:
            chn.env_release = decay_coeff(e[3])
        chn.sweep_pitch = self.sweep_pitch
        if self.portamento:
            chn.sweep_pitch = s16(chn.sweep_pitch + ((self.porta_key - key) << 6))
        if self.porta_time == 0:
            chn.sweep_length = length
            chn.auto_sweep = False
        else:
            chn.sweep_length = (self.porta_time * self.porta_time * abs(chn.sweep_pitch)) >> 11
        chn.sweep_counter = 0
        if new and vel > 0:
            n = Note()
            n.track, n.key, n.vel = self.idx, key, vel
            n.patch = self.program
            n.adsr = tuple(e)
            n.start = float(p.tick_now)
            n.end = None
            n.why = None
            p.serial += 1
            n.serial = chn.serial = p.serial
            chn.note = n
            p.notes.append(n)
        elif new:
            p.serial += 1
            chn.serial = p.serial
        self.note_tick = p.tick_now
        if not self.note_ticks or self.note_ticks[-1] != p.tick_now:
            self.note_ticks.append(p.tick_now)

    def loop_back(self, target_tick):
        """A jump back to code first run at target_tick. A loop that strikes no
        note is an idle track (a keep-alive: wait, jump to itself) — it is not
        the song's loop."""
        j = self.p.tick_now
        silent = not any(target_tick <= n < j for n in self.note_ticks[-64:])
        self.loops.append((target_tick, j, silent))

    def idle(self):
        return bool(self.loops) and self.loops[-1][2]

    # --- TrackStepTicks ----------------------------------------------------------
    def step(self):
        p = self.p
        for chn in self.chans:
            if chn.length > 0:
                chn.length -= 1
            if not chn.auto_sweep and chn.sweep_counter < chn.sweep_length:
                chn.sweep_counter += 1
        if self.note_finish_wait:
            if self.chans:
                return 0
            self.note_finish_wait = False
        if self.wait > 0:
            self.wait -= 1
            if self.wait > 0:
                return 0
        guard = 0
        while self.wait == 0 and not self.note_finish_wait:
            guard += 1
            if guard > 50000:
                p.warn(f'track {self.idx}: a loop with no wait in it at {self.cur:#x}; stopped')
                return -1
            at = self.cur
            if at not in self.visited:
                self.visited[at] = p.tick_now
            special = None
            run = True
            cmd = self.r8()
            if cmd == 0xA2:
                cmd = self.r8()
                run = self.cmp
            if cmd == 0xA0:
                cmd = self.r8()
                special = 'ran'
            if cmd == 0xA1:
                cmd = self.r8()
                special = 'var'
            if not (cmd & 0x80):
                vel = self.r8()
                length = self.value(special or 'vlv')
                key = cmd + self.transpose
                if not run:
                    continue
                key = min(max(key, 0), 127)
                if not self.muted:
                    self.play_note(key, min(vel, 127), length if length > 0 else -1)
                self.porta_key = key
                if self.note_wait:
                    self.wait = length
                    if length == 0:
                        self.note_finish_wait = True
                continue
            hi = cmd & 0xF0
            if hi == 0x80:
                par = self.value(special or 'vlv')
                if run:
                    if cmd == 0x80:
                        self.wait = par
                    elif cmd == 0x81 and par < 0x10000:
                        self.program = par
            elif hi == 0x90:
                if cmd == 0x93:
                    num = self.r8()
                    off = self.r24()
                    if run:
                        t = p.get_track(num)
                        if t is not None and t is not self:
                            p.track_stop(t, 'track-restart')
                            t.base, t.cur = 0, off
                elif cmd == 0x94:
                    off = self.r24()
                    if run:
                        if off in self.visited:
                            self.loop_back(self.visited[off])
                        self.cur = off
                elif cmd == 0x95:
                    off = self.r24()
                    if run and len(self.stack) < 3:
                        self.stack.append([self.cur, 0])
                        self.cur = off
            elif hi in (0xC0, 0xD0):
                v = self.value(special or 'u8') & 0xFF
                if not run:
                    continue
                if cmd == 0xC1:
                    self.volume = v
                elif cmd == 0xD5:
                    self.expression = v
                elif cmd == 0xC2:
                    p.volume = v
                elif cmd == 0xC5:
                    self.bend_range = v
                elif cmd == 0xC6:
                    self.priority = v
                elif cmd == 0xC7:
                    self.note_wait = bool(v)
                elif cmd == 0xCF:
                    self.porta_time = v
                elif cmd == 0xCA:
                    self.mod.depth = v
                elif cmd == 0xCB:
                    self.mod.speed = v
                elif cmd == 0xCC:
                    self.mod.target = v
                elif cmd == 0xCD:
                    self.mod.range = v
                elif cmd in (0xD0, 0xD1, 0xD2, 0xD3):
                    self.env[cmd - 0xD0] = v
                elif cmd == 0xD4:
                    if len(self.stack) < 3:
                        self.stack.append([self.cur, v])
                elif cmd == 0xC8:
                    self.tie = bool(v)
                    p.track_release(self, -1, 'tie')
                    p.track_free(self)
                elif cmd == 0xD7:
                    p.track_mute(self, v)
                elif cmd == 0xC9:
                    self.porta_key = (v + self.transpose) & 0xFF
                    self.portamento = True
                elif cmd == 0xCE:
                    self.portamento = bool(v)
                elif cmd == 0xC3:
                    self.transpose = s8(v)
                elif cmd == 0xC4:
                    self.pitch_bend = s8(v)
                elif cmd == 0xC0:
                    self.pan = s8(v - 0x40)
            elif hi == 0xE0:
                v = s16(self.value(special or 'u16'))
                if not run:
                    continue
                if cmd == 0xE3:
                    self.sweep_pitch = v
                elif cmd == 0xE1:
                    p.tempo = v & 0xFFFF
                    p.tempo_log.append((float(p.tick_now), p.tempo))
                elif cmd == 0xE0:
                    self.mod.delay = v & 0xFFFF
            elif hi == 0xB0:
                vn = self.r8()
                v = s16(self.value(special or 'u16'))
                if not run or not (0 <= vn < 32):
                    continue
                a = p.variables[vn]
                if cmd == 0xB0:
                    a = v
                elif cmd == 0xB1:
                    a += v
                elif cmd == 0xB2:
                    a -= v
                elif cmd == 0xB3:
                    a *= v
                elif cmd == 0xB4:
                    if v != 0:
                        a = tdiv(a, v)
                elif cmd == 0xB5:
                    a = a << v if v >= 0 else a >> -v
                elif cmd == 0xB6:
                    neg = v < 0
                    if neg:
                        v = s16(-v)
                    r = (p.rand() * (v + 1)) >> 16
                    a = -r if neg else r
                elif cmd == 0xB8:
                    self.cmp = a == v
                elif cmd == 0xB9:
                    self.cmp = a >= v
                elif cmd == 0xBA:
                    self.cmp = a > v
                elif cmd == 0xBB:
                    self.cmp = a <= v
                elif cmd == 0xBC:
                    self.cmp = a < v
                elif cmd == 0xBD:
                    self.cmp = a != v
                p.variables[vn] = s16(a)
            elif hi == 0xF0:
                if not run:
                    continue
                if cmd == 0xFD:
                    if self.stack:
                        self.cur = self.stack.pop()[0]
                elif cmd == 0xFC:
                    if not self.stack:
                        continue
                    top = self.stack[-1]
                    lc = top[1]
                    if lc != 0:
                        lc -= 1
                        if lc == 0:
                            self.stack.pop()
                            continue
                    elif top[0] in self.visited:            # an endless loop
                        self.loop_back(self.visited[top[0]])
                    top[1] = lc
                    self.cur = top[0]
                elif cmd == 0xFF:
                    return -1
        return 0


class Player:
    def __init__(self, seq_data, bank_entries, swars, channel_mask=0xFFFF,
                 sseq_vol=127, prio=64, warn=print, vibrato=True):
        self.data = seq_data
        self.bank = bank_entries
        self.swars = swars
        self.prio = prio
        self.warn = warn
        self.vibrato = vibrato
        self.channels = [ExChannel(i) for i in range(16)]
        self.rand_state = 0x12345678
        self.variables = [-1] * 32
        # PlayerInit
        self.tempo = 120
        self.tempo_ratio = 256
        self.tempo_counter = TIMER_RATE
        self.volume = 0x7F
        self.ext_fader = cnv_scale(sseq_vol)
        self.tick_now = 0
        self.ticks_done = 0
        self.frame = 0
        self.serial = 0
        self.dropped = 0
        self.notes = []
        self.tempo_log = [(0.0, 120)]
        self.curves = {}
        self.active = True
        # SND_PrepareSeq: track 0 at the start; 0xFE allocates the others
        self.tracks = [None] * 16
        t0 = Track(self, 0)
        t0.base, t0.cur = 0, 0
        self.tracks[0] = t0
        if self.data and self.data[0] == 0xFE:
            mask = (self.data[1] | (self.data[2] << 8)) >> 1
            t0.cur = 3
            n = 1
            while mask:
                if mask & 1:
                    self.tracks[n] = Track(self, n)
                n += 1
                mask >>= 1
        for t in self.tracks:
            if t is not None:
                t.channel_mask = channel_mask

    # --- helpers ---------------------------------------------------------------
    def rand(self):
        self.rand_state = (self.rand_state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.rand_state >> 16

    def var_get(self, n):
        return self.variables[n] if 0 <= n < 32 else 0

    def get_track(self, n):
        t = self.tracks[n] if 0 <= n < 16 else None
        return None if t is None or t.stopped else t

    def pos_frame(self):
        return max(0.0, self.ticks_done - 1 + self._c_after / TIMER_RATE)

    def read_inst(self, program, key):                 # SND_ReadInstData
        if program < 0 or program >= len(self.bank):
            return None
        e = self.bank[program]
        if e.record in (INST_PCM, INST_PSG, INST_NOISE, INST_DIRECTPCM, INST_DUMMY):
            return (e.record, e.regions[0]) if e.regions else None
        if e.record == 16:
            if not e.regions or key < e.regions[0].lo or key > e.regions[-1].hi:
                return None
            r = e.regions[key - e.regions[0].lo]
            return r.record, r
        if e.record == 17:
            for r in e.regions:
                if key <= r.hi:
                    return r.record, r
            return None
        return None

    def wave(self, region):
        sw = self.swars[region.swar] if region.swar < len(self.swars) else None
        return sw.swav(region.swav) if sw is not None else None

    def note_on(self, chn, key, vel, length, itype, region):   # SND_NoteOn
        release = region.r
        if release == 0xFF:
            length = -1
            release = 0
        if itype == INST_PCM:
            w = self.wave(region)
            if w is None:
                return False
            chn.wave_timer = w.timer
            chn.wave_len = 0 if w.loop else len(w.pcm)
        elif itype == INST_PSG:
            if not (8 <= chn.id <= 13):
                return False
            chn.wave_timer = PSG_TIMER
            chn.wave_len = 0
        elif itype == INST_NOISE:
            if not (14 <= chn.id <= 15):
                return False
            chn.wave_timer = PSG_TIMER
            chn.wave_len = 0
        else:
            return False
        chn.kind = itype
        # ExChannelStart
        chn.env_att = -92544
        chn.env_status = ENV_ATTACK
        chn.length = length
        chn.lfo.counter = chn.lfo.delay_counter = 0
        chn.start = True
        chn.active = True
        chn.hw_active = False
        chn._sync_start = False
        chn.play_pos = 0.0
        chn.midi_key = key
        chn.root_key = region.note
        chn.velocity = vel
        chn.env_attack = attack_coeff(region.a)
        chn.env_decay = decay_coeff(region.d)
        chn.env_sustain = region.s
        chn.env_release = decay_coeff(release)
        chn.init_pan = s8(region.pan - 0x40)
        return True

    def end_note(self, chn, pos, why):
        if chn.note is not None and chn.note.end is None:
            chn.note.end = pos
            chn.note.why = why
        chn.note = None

    def unlink(self, chn):                              # ChannelCallback
        t = chn.track
        if t is not None and chn in t.chans:
            t.chans.remove(chn)
        chn.track = None

    def alloc_channel(self, mask, priority, track):     # SND_AllocExChannel
        prev = None
        for n in ALLOC_ORDER:
            if not (mask >> n) & 1:
                continue
            c = self.channels[n]
            if prev is None:
                prev = c
                continue
            if c.priority > prev.priority:
                continue
            if c.priority != prev.priority or self._vol_cmp(prev, c) < 0:
                prev = c
        if prev is None or priority < prev.priority:
            return None
        if prev.active and prev.note is not None:
            self.end_note(prev, float(self.tick_now), 'stolen')
        self.unlink(prev)
        prev.note = None
        prev.active = False
        prev.hw_active = False
        prev.setup(priority)
        return prev

    @staticmethod
    def _vol_cmp(a, b):
        va = ((a.volume & 0xFF) << 4) >> SHIFT[(a.volume >> 8) & 3]
        vb = ((b.volume & 0xFF) << 4) >> SHIFT[(b.volume >> 8) & 3]
        if va != vb:
            return 1 if va < vb else -1
        return 0

    def release(self, chn, pos, why):
        chn.priority = 1
        chn.env_status = ENV_RELEASE
        self.end_note(chn, pos, why)

    def track_update(self, t, release):                 # TrackUpdateChannel
        vol = DB[t.volume] + DB[t.expression] + DB[self.volume]
        fader = self.ext_fader
        pitch = (t.pitch_bend * (t.bend_range << 6)) >> 7
        pan = t.pan
        if t.pan_range != 127:
            pan = (pan * t.pan_range + 0x40) >> 7
        vol = max(vol, -0x8000)
        pan = min(max(pan, -128), 127)
        for c in t.chans:
            c.user_decay2 = fader
            if c.env_status == ENV_RELEASE:
                continue
            c.user_decay = vol
            c.user_pitch = pitch
            c.user_pan = pan
            c.pan_range = t.pan_range
            c.lfo.param_from(t.mod)
            if c.length == 0 and release:
                self.release(c, float(int(self.pos_frame())), 'length')
        return vol + fader, pan

    def track_release(self, t, release, why):           # TrackReleaseChannels
        self.track_update(t, 0)
        for c in t.chans:
            if c.active:
                if release >= 0:
                    c.env_release = decay_coeff(release & 0xFF)
                self.release(c, float(self.tick_now), why)

    def track_free(self, t):                            # TrackFreeChannels
        for c in t.chans:
            c.track = None
        t.chans = []

    def track_stop(self, t, why):
        self.track_release(t, -1, why)
        self.track_free(t)

    def track_mute(self, t, mode):
        if mode == 0:
            t.muted = False
        elif mode == 1:
            t.muted = True
        elif mode == 2:
            t.muted = True
            self.track_release(t, -1, 'mute')
        elif mode == 3:
            t.muted = True
            self.track_release(t, 127, 'mute')
            self.track_free(t)

    # --- a frame ------------------------------------------------------------------
    def frame_step(self):
        # SND_UpdateExChannel: channels started last frame start on the hardware
        for c in self.channels:
            if c.active and c.start is False and not c.hw_active and c._sync_start:
                c.hw_active = True
                c._sync_start = False
        # SND_SeqMain → PlayerSeqMain
        for t in self.tracks:
            if t is not None:
                t.note_tick = None
        if self.active:
            ticks = 0
            while self.tempo_counter >= TIMER_RATE:
                self.tempo_counter -= TIMER_RATE
                ticks += 1
            for _ in range(ticks):
                self.tick_now = self.ticks_done
                playing = False
                for i in range(16):
                    t = self.tracks[i]
                    if t is not None and t.cur is not None and not t.stopped:
                        if t.step():
                            self.track_stop(t, 'track-end')
                            t.stopped = True
                            t.stop_tick = self.tick_now
                        else:
                            playing = True
                self.ticks_done += 1
                if not playing:
                    self.active = False
                    break
            self._c_after = self.tempo_counter
            self.tempo_counter += (self.tempo * self.tempo_ratio) >> 8
        else:
            self._c_after = 0
        # PlayerUpdateChannel
        levels = {}
        for t in self.tracks:
            if t is not None and t.cur is not None and not t.stopped:
                levels[t.idx] = self.track_update(t, 1)
        # SND_ExChannelMain
        for c in self.channels:
            self.ex_channel_main(c)
        self.sample_curves(levels)
        self.rand()
        self.frame += 1

    def ex_channel_main(self, c):
        if not c.active:
            return
        if c.start:
            c.start = False
            c._sync_start = True
        elif not c.hw_active:
            self.end_note(c, self.pos_frame(), 'sample-end')
            self.unlink(c)
            c.priority = 0
            c.volume = 0
            c.active = False
            return
        vol = DB[c.velocity & 0x7F] if c.velocity < 128 else DB[127]
        pitch = (c.midi_key - c.root_key) * 0x40
        # SND_UpdateExChannelEnvelope (step)
        if c.env_status == ENV_ATTACK:
            c.env_att = -((-c.env_att * c.env_attack) >> 8)
            if c.env_att == 0:
                c.env_status = ENV_DECAY
        elif c.env_status == ENV_DECAY:
            sus = DB[c.env_sustain & 0x7F] << 7
            c.env_att -= c.env_decay
            if c.env_att <= sus:
                c.env_att = sus
                c.env_status = ENV_SUSTAIN
        elif c.env_status == ENV_RELEASE:
            c.env_att -= c.env_release
        vol += c.env_att >> 7
        # ExChannelSweepUpdate
        sweep = 0
        if c.sweep_pitch != 0 and c.sweep_counter < c.sweep_length:
            sweep = tdiv(c.sweep_pitch * (c.sweep_length - c.sweep_counter), c.sweep_length)
            if c.auto_sweep:
                c.sweep_counter += 1
        pitch += sweep
        vol += c.user_decay + c.user_decay2
        pitch += c.user_pitch
        # ExChannelLfoUpdate
        lf = c.lfo
        lfo = 0
        if lf.depth and lf.delay_counter >= lf.delay:
            lfo = sin_idx((lf.counter >> 8) & 0x7F) * lf.depth * lf.range
            if lfo:
                if lf.target == LFO_VOLUME:
                    lfo *= 60
                else:
                    lfo <<= 6
                lfo >>= 14
        if lf.delay_counter < lf.delay:
            lf.delay_counter += 1
        else:
            tmp = (lf.counter + (lf.speed << 6)) >> 8
            while tmp >= 0x80:
                tmp -= 0x80
            lf.counter = ((lf.counter + (lf.speed << 6)) & 0xFF) | (tmp << 8)
        pan = 0
        c.lfo_vol = c.lfo_pan = 0
        vib = 0
        if lf.target == LFO_VOLUME:
            if vol > -0x8000:
                vol += lfo
            c.lfo_vol = lfo
        elif lf.target == LFO_PAN:
            pan += lfo
            c.lfo_pan = lfo
        elif lf.target == LFO_PITCH:
            pitch += lfo
            vib = lfo
        pan += c.init_pan
        if c.pan_range != 127:
            pan = (pan * c.pan_range + 0x40) >> 7
        pan += c.user_pan
        ref = c.note.key if c.note is not None else c.midi_key
        c.adj = pitch - (c.midi_key - c.root_key) * 0x40 + (c.midi_key - ref) * 0x40 \
            - (0 if self.vibrato else vib)
        if c.env_status == ENV_RELEASE and vol <= -723:
            self.end_note(c, self.pos_frame(), 'release-end')
            self.unlink(c)
            c.priority = 0
            c.volume = 0
            c.active = False
            c.hw_active = False
            return
        c.volume = calc_channel_volume(vol)
        timer = calc_timer(c.wave_timer, pitch)
        if c.kind == INST_PSG:
            timer &= 0xFFFC
        c.timer = timer
        if c.hw_active and c.wave_len:
            c.play_pos += ARM7_CLOCK / 2.0 / max(timer, 1) * FRAME_SEC
            if c.play_pos >= c.wave_len:
                c.hw_active = False          # the hardware stops; freed next frame

    def governing(self, t):
        best = None
        for c in t.chans:
            k = (c.env_status != ENV_RELEASE, c.serial)
            if best is None or k > best[0]:
                best = (k, c)
        return best[1] if best else None

    def sample_curves(self, levels):
        pre = self.pos_frame()
        for idx, (level, pan) in levels.items():
            t = self.tracks[idx]
            pos = float(t.note_tick) if t.note_tick is not None else pre
            c = self.governing(t)
            bend = (t.pitch_bend * (t.bend_range << 6)) >> 7
            lvol = lpan = 0
            if c is not None:
                bend = c.adj
                lvol, lpan = c.lfo_vol, c.lfo_pan
                if c.note is not None and c.velocity != c.note.vel:
                    lvol += DB[min(c.velocity, 127)] - DB[min(c.note.vel, 127)]
            rec = (bend, level, pan, lvol, lpan)
            cl = self.curves.setdefault(idx, [])
            if not cl or cl[-1][1:] != rec:
                cl.append((pos,) + rec)

    # --- whole song -----------------------------------------------------------------
    def live_tracks(self):
        return [t for t in self.tracks if t is not None and t.cur is not None and not t.stopped]

    def music_tracks(self):
        """Running tracks that still play: not stopped, not idling."""
        return [t for t in self.live_tracks() if not t.idle()]

    def loop_window(self):
        """Once every playing track has jumped back with notes in its loop:
        (start, end, lengths) of a window over which the whole song repeats —
        starting no earlier than the last note of any track that has ended or
        gone idle, so a part that does not loop is never cut short."""
        live = self.music_tracks()
        if not live:
            return None
        starts, lens = [], []
        for t in live:
            loops = [l for l in t.loops if not l[2]]
            if not loops:
                return None
            s, j, _ = loops[0]
            starts.append(s)
            lens.append(j - s)
        if any(l <= 0 for l in lens):
            return None
        from math import gcd
        L = 1
        for l in lens:
            L = L * l // gcd(L, l)
        S = max(starts)
        done = [t.note_ticks[-1] for t in self.tracks
                if t is not None and t not in live and t.note_ticks]
        if done and max(done) >= S:
            S += ((max(done) - S) // L + 1) * L
        return S, S + L, lens


ExChannel._sync_start = False


def simulate(seq_data, bank_entries, swars, channel_mask, sseq_vol, prio,
             max_seconds=900.0, warn=print, vibrato=True):
    """Run the song. → (player, loop): loop = (start_tick, end_tick) or None;
    player.end_tick is where the song (or its first pass of the loop) ends."""
    p = Player(seq_data, bank_entries, swars, channel_mask, sseq_vol, prio, warn, vibrato)
    max_frames = int(max_seconds / FRAME_SEC)
    loop = None
    while p.frame < max_frames:
        p.frame_step()
        if loop is None:
            w = p.loop_window()
            if w is not None:
                S, E, lens = w
                if E - S > 16 * max(lens):
                    warn(f'loop lengths {lens} have an unwieldy common multiple; looping on the longest')
                    E = S + max(lens)
                loop = (S, E)
        if loop is not None and p.ticks_done > loop[1]:
            break
        if loop is None and not p.music_tracks() and not any(c.active for c in p.channels):
            break
    else:
        warn(f'stopped after {max_seconds:.0f} s without a loop or an end')
    end = loop[1] if loop else max(1, int(p.pos_frame()) + 1)
    for c in p.channels:
        if c.note is not None:
            p.end_note(c, float(end), 'song-end')
    p.end_tick = end
    p.order = [t.idx for t in p.tracks if t is not None]
    return p, loop
