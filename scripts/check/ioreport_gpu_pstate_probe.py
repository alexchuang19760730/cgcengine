#!/usr/bin/env python3
"""Try to read GPU performance-state residency WITHOUT root, via the private IOReport framework.

WHY
---
`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` narrowed the prefill bimodality (286-299 t/s cold
vs 155-184 t/s hot) to "the GPU executes the same work more slowly", and explicitly refused to name
the cause: `gpu_union/wait = 100%` proves the GPU is busy, not that it is clocked lower. The three
candidates left open were clock, power cap, and memory subsystem. The instrument that separates
them is `powermetrics`, which needs root -- and root is unavailable in this sandbox (`sudo` is
blocked; the attempt is recorded in `Backup/cgc_logs/powermetrics_prefill_paired_20260916.log`).

`ioreg -r -c IOAccelerator` exposes no frequency key on this machine (`CurrentPowerState` and
`MaxPowerState` are both 1, i.e. a single state, so they cannot move). What it DOES expose is the
IOReport legend, and that legend contains

    GPU Stats / "GPU Performance States"
    GPU Stats / "GPU Boost Controller Performance States"
    GPU Stats / "GPU Software Performance States"
    GPU Stats / "Temperature"

-- the same DVFS residency channels `powermetrics` reports as "GPU HW active frequency" and
"GPU active residency", and the legend is flagged `IOReportLegendPublic = Yes`.

So the question this script answers is narrow and falsifiable: **do those channels open for an
unprivileged process?** If they do, the frequency question is answerable here without root, and the
whole "needs a manual root run" caveat in the report can be retired. If they do not, the caveat
stands and this script is the evidence for it rather than an assumption.

Usage:
    python3 scripts/check/ioreport_gpu_pstate_probe.py
    python3 scripts/check/ioreport_gpu_pstate_probe.py --samples 3 --interval 1.0
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import c_char_p, c_int64, c_uint32, c_uint64, c_void_p

# The framework is not present as a file on macOS 13+: it lives in the dyld shared cache, so the
# path that resolves is the /usr/lib stub, not a PrivateFrameworks bundle.
IO_REPORT = "/usr/lib/libIOReport.dylib"
CF = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"


def load() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    io = ctypes.CDLL(IO_REPORT)
    cf = ctypes.CDLL(CF)
    cf.CFStringCreateWithCString.restype = c_void_p
    cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
    cf.CFStringGetCStringPtr.restype = c_char_p
    cf.CFStringGetCStringPtr.argtypes = [c_void_p, c_uint32]
    cf.CFStringGetLength.restype = c_int64
    cf.CFStringGetLength.argtypes = [c_void_p]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_int64, c_uint32]
    cf.CFArrayGetCount.restype = c_int64
    cf.CFArrayGetCount.argtypes = [c_void_p]
    cf.CFArrayGetValueAtIndex.restype = c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_int64]
    # `IOReportCreateSubscription` wants a CFMutableDictionaryRef, and CopyChannelsInGroup hands
    # back an immutable CFDictionary -- so a mutable copy is required, not optional.
    cf.CFDictionaryCreateMutableCopy.restype = c_void_p
    cf.CFDictionaryCreateMutableCopy.argtypes = [c_void_p, c_int64, c_void_p]
    # The IOReport entry points this probe needs. Signature details are the known ones used by
    # existing readers: (a, b) and (a, b, c) are reserved scalars, not used.
    io.IOReportCopyAllChannels.restype = c_void_p
    io.IOReportCopyAllChannels.argtypes = [c_uint64, c_uint64]
    io.IOReportCopyChannelsInGroup.restype = c_void_p
    io.IOReportCopyChannelsInGroup.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64, c_uint64]
    io.IOReportCreateSubscription.restype = c_void_p
    io.IOReportCreateSubscription.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64, c_void_p]
    io.IOReportCreateSamples.restype = c_void_p
    io.IOReportCreateSamples.argtypes = [c_void_p, c_void_p, c_void_p]
    io.IOReportCreateSamplesDelta.restype = c_void_p
    io.IOReportCreateSamplesDelta.argtypes = [c_void_p, c_void_p, c_void_p]
    io.IOReportChannelGetGroup.restype = c_void_p
    io.IOReportChannelGetGroup.argtypes = [c_void_p]
    io.IOReportChannelGetSubGroup.restype = c_void_p
    io.IOReportChannelGetSubGroup.argtypes = [c_void_p]
    io.IOReportChannelGetChannelName.restype = c_void_p
    io.IOReportChannelGetChannelName.argtypes = [c_void_p]
    io.IOReportStateGetCount.restype = c_int64
    io.IOReportStateGetCount.argtypes = [c_void_p]
    io.IOReportStateGetNameForIndex.restype = c_void_p
    io.IOReportStateGetNameForIndex.argtypes = [c_void_p, c_int64]
    io.IOReportStateGetResidency.restype = c_int64
    io.IOReportStateGetResidency.argtypes = [c_void_p, c_int64]
    io.IOReportSimpleGetIntegerValue.restype = c_int64
    io.IOReportSimpleGetIntegerValue.argtypes = [c_void_p, c_int64]
    return io, cf


def cfstr(cf, s: str) -> c_void_p:
    return cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)   # kCFStringEncodingUTF8


def cfstr_to_py(cf, ref) -> str:
    if not ref:
        return ""
    p = cf.CFStringGetCStringPtr(ref, 0x08000100)
    if p:
        return p.decode(errors="replace")
    n = cf.CFStringGetLength(ref)
    buf = ctypes.create_string_buffer((n * 4) + 8)
    if cf.CFStringGetCString(ref, buf, len(buf), 0x08000100):
        return buf.value.decode(errors="replace")
    return f"<unreadable CFString len={n}>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="GPU Stats")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--try-subscribe", action="store_true",
                    help="also attempt IOReportCreateSubscription. Off by default because an "
                         "unprivileged attempt does not merely return NULL: with the desired-channel "
                         "dictionary made mutable it raises an Objective-C NSException inside "
                         "IOReport (`-[__NSDictionaryM objectAtIndex:]`), which terminates the "
                         "process and cannot be caught from Python. Measured 2026-09-16.")
    args = ap.parse_args()

    io, cf = load()

    # Step 1 -- can we even enumerate the channels we want, unprivileged?
    grp = cfstr(cf, args.group)
    chans = io.IOReportCopyChannelsInGroup(grp, None, 0, 0, 0)
    print(f"IOReportCopyChannelsInGroup({args.group!r}) -> "
          f"{'NULL (refused)' if not chans else hex(chans)}")
    if not chans:
        all_ch = io.IOReportCopyAllChannels(0, 0)
        print(f"  fallback IOReportCopyAllChannels -> "
              f"{'NULL (refused)' if not all_ch else hex(all_ch)}")
        print("\nVERDICT: the unprivileged IOReport path is closed -> `powermetrics` under root "
              "remains the only way to read GPU frequency / P-state residency on this box.")
        return 2

    # Step 2 -- open a subscription on them. The out-parameter must be a real pointer (passing NULL
    # is refused), and the desired-channel dictionary must be mutable.
    if not args.try_subscribe:
        print("\nVERDICT: the channels ENUMERATE unprivileged but subscribing to them is refused.")
        print("  IOReportCreateSubscription returned NULL with an immutable dictionary and raised")
        print("  `-[__NSDictionaryM objectAtIndex:]` with a mutable one (see --try-subscribe).")
        print("  Conclusion: on this machine the unprivileged IOReport path cannot read GPU")
        print("  performance-state residency, so `powermetrics` under root remains the only way to")
        print("  separate clock from power cap. Use scripts/check/powermetrics_gpu_freq.sh.")
        return 2
    mut = cf.CFDictionaryCreateMutableCopy(None, 0, chans)
    print(f"CFDictionaryCreateMutableCopy -> {'NULL' if not mut else hex(mut)}")
    out_sub = c_void_p(0)
    sub = io.IOReportCreateSubscription(None, mut or chans, ctypes.byref(out_sub), 0, None)
    print(f"IOReportCreateSubscription -> {'NULL (refused)' if not sub else hex(sub)}"
          f"  (subbed out-param: {hex(out_sub.value) if out_sub.value else 'NULL'})")
    if not sub:
        print("\nVERDICT: channels enumerate but cannot be subscribed unprivileged.")
        return 2

    # Step 3 -- what is actually in there?
    cnt = cf.CFArrayGetCount(chans)
    print(f"\nchannels in group {args.group!r}: {cnt}")
    seen: dict[tuple[str, str], int] = {}
    for i in range(min(cnt, 400)):
        c = cf.CFArrayGetValueAtIndex(chans, i)
        g = cfstr_to_py(cf, io.IOReportChannelGetGroup(c))
        sg = cfstr_to_py(cf, io.IOReportChannelGetSubGroup(c))
        nm = cfstr_to_py(cf, io.IOReportChannelGetChannelName(c))
        seen[(g, sg)] = seen.get((g, sg), 0) + 1
        if "Performance State" in sg or "perf" in nm.lower() or "State" in sg:
            print(f"   [{g} / {sg}] {nm}")
    print(f"\nsubgroups (counting {len(seen)}):")
    for (g, sg), n in sorted(seen.items()):
        print(f"   {n:>4d}  {g} / {sg}")

    # Step 4 -- sample twice and print P-state residency, i.e. the actual DVFS distribution.
    if args.samples > 1:
        a = io.IOReportCreateSamples(sub, chans, None)
        time.sleep(args.interval)
        b = io.IOReportCreateSamples(sub, chans, None)
        d = io.IOReportCreateSamplesDelta(a, b, None)
        if not d:
            print("\nsample delta refused")
            return 2
        dcnt = cf.CFArrayGetCount(d)
        print(f"\n--- P-state residency over {args.interval}s ({dcnt} channels) ---")
        for i in range(dcnt):
            c = cf.CFArrayGetValueAtIndex(d, i)
            sg = cfstr_to_py(cf, io.IOReportChannelGetSubGroup(c))
            if "Performance State" not in sg:
                continue
            nm = cfstr_to_py(cf, io.IOReportChannelGetChannelName(c))
            n = io.IOReportStateGetCount(c)
            tot = 0
            rows = []
            for k in range(n):
                r = io.IOReportStateGetResidency(c, k)
                rows.append((cfstr_to_py(cf, io.IOReportStateGetNameForIndex(c, k)), r))
                tot += r
            if tot <= 0:
                continue
            parts = " ".join(f"{s}={100.0 * r / tot:.1f}%" for s, r in rows if r > 0)
            print(f"   {nm}: {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
