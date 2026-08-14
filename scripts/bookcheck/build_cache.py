"""Build a feature cache once: rsi, dd252, mom(12-1), close/open arrays, date index."""
import json, os, pickle, sys
from collections import deque

ROOT = "/home/user/equity-conviction-monitor"
H = os.path.join(ROOT, "ledger", "history")
ETFS = {"SPY", "RSP", "IWB", "IWV", "IWM", "VTI"}
OUT = "/tmp/claude-0/-home-user-equity-conviction-monitor/58c3f77b-53ed-5f51-98fd-1c74f891c4af/scratchpad/feat.pkl"


def rsi_series(c, w=14):
    out = [None] * len(c)
    if len(c) < w + 1:
        return out
    g = l_ = 0.0
    for i in range(1, w + 1):
        ch = c[i] - c[i - 1]
        g += max(ch, 0.0); l_ += max(-ch, 0.0)
    ag, al = g / w, l_ / w
    out[w] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(w + 1, len(c)):
        ch = c[i] - c[i - 1]
        ag = (ag * (w - 1) + max(ch, 0.0)) / w
        al = (al * (w - 1) + max(-ch, 0.0)) / w
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def rolling_max(v, w):
    dq, out = deque(), [0.0] * len(v)
    for i, x in enumerate(v):
        while dq and v[dq[-1]] <= x: dq.pop()
        dq.append(i)
        if dq[0] <= i - w: dq.popleft()
        out[i] = v[dq[0]]
    return out


def rolling_mean_sd(vals, window):
    s = ss = 0.0
    means = [None]*len(vals); sds = [None]*len(vals)
    for i, v in enumerate(vals):
        s += v; ss += v*v
        if i >= window:
            s -= vals[i-window]; ss -= vals[i-window]**2
        if i >= window-1:
            m = s/window; means[i]=m; sds[i]=max(ss/window - m*m, 0.0)**0.5
    return means, sds


def rolling_atr(highs, lows, closes, window=14):
    out=[None]*len(closes); trs=[0.0]*len(closes)
    for i in range(1,len(closes)):
        pc=closes[i-1]
        trs[i]=max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc))
    s=0.0
    for i in range(1,len(closes)):
        s+=trs[i]
        if i>window: s-=trs[i-window]
        if i>=window: out[i]=s/window
    return out


def rolling_beta(rs, rb, window=252):
    n=len(rs); out=[None]*n
    sxy=sxx=sx=sy=0.0
    for i in range(1,n):
        sxy+=rs[i]*rb[i]; sxx+=rb[i]*rb[i]; sx+=rb[i]; sy+=rs[i]
        if i>window:
            j=i-window
            sxy-=rs[j]*rb[j]; sxx-=rb[j]*rb[j]; sx-=rb[j]; sy-=rs[j]
        if i>=window:
            k=window
            cov=sxy-sx*sy/k; var=sxx-sx*sx/k
            out[i]=(cov/var) if var>1e-12 else None
    return out


def main():
    raw = {}
    for fn in sorted(os.listdir(H)):
        if not fn.endswith(".json"): continue
        d = json.load(open(os.path.join(H, fn)))
        if len(d.get("close") or []) >= 320:
            raw[fn[:-5]] = d
    bench = raw["RSP"]
    bidx = {d: i for i, d in enumerate(bench["dates"])}
    bc = bench["close"]
    brets = [0.0]*len(bc)
    for i in range(1,len(bc)): brets[i]=bc[i]/bc[i-1]-1.0
    spy = raw["SPY"]
    delisted = {s for s,d in raw.items() if d["dates"][-1] < "2026-08-01"}
    feats = {}
    for sym, d in raw.items():
        if sym in ETFS: continue
        c,h,l,o = d["close"], d["high"], d["low"], d["open"]
        dates = d["dates"]
        n=len(c)
        hi252 = rolling_max(h,252)
        ma50, sd50 = rolling_mean_sd(c,50)
        ma200,_ = rolling_mean_sd(c,200)
        # align benchmark returns to this name's calendar for beta
        rs=[0.0]*n; rb=[0.0]*n
        for i in range(1,n):
            rs[i]= c[i]/c[i-1]-1.0 if c[i-1]>0 else 0.0
            bi=bidx.get(dates[i]); bj=bidx.get(dates[i-1])
            rb[i]= (bc[bi]/bc[bj]-1.0) if (bi is not None and bj is not None) else 0.0
        feats[sym] = {
            "dates": dates, "close": c, "open": o, "high": h, "low": l,
            "vol": d.get("volume"),
            "rsi": rsi_series(c),
            "dd252": [(hi252[i]-c[i])/hi252[i] if hi252[i]>0 else 0.0 for i in range(n)],
            "atr": rolling_atr(h,l,c),
            "ma50": ma50, "sd50": sd50, "ma200": ma200,
            "mom": [((c[i-21]/c[i-252]-1.0) if i>=252 and c[i-252]>0 else None) for i in range(n)],
            "beta": rolling_beta(rs, rb),
            "idx": {dt_: i for i, dt_ in enumerate(dates)},
            "delisted": sym in delisted,
        }
    pickle.dump({"feats":feats, "bdates":bench["dates"], "bclose":bc, "bidx":bidx,
                 "spy_dates":spy["dates"], "spy_close":spy["close"]}, open(OUT,"wb"), 4)
    print("names", len(feats), "delisted-in-panel", sum(1 for f in feats.values() if f['delisted']))

main()
