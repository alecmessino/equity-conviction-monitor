"""Precompute raw daily signal lists per screen, with a tradeability filter."""
import pickle, os, sys
SP="/tmp/claude-0/-home-user-equity-conviction-monitor/58c3f77b-53ed-5f51-98fd-1c74f891c4af/scratchpad/"
D=pickle.load(open(SP+"feat.pkl","rb"))
F=D["feats"]

def _oversold(f,i,rsi_max=35.0,z_max=-1.5):
    r,ma,sd=f["rsi"][i],f["ma50"][i],f["sd50"][i]
    z=(f["close"][i]-ma)/sd if (ma is not None and sd) else None
    return (r is not None and r<=rsi_max) or (z is not None and z<=z_max)

def _rr(f,i,fib=0.382):
    a=f["atr"][i]
    if not a: return 0.0
    hi=f["close"][i]/(1-f["dd252"][i]) if f["dd252"][i]<1 else f["close"][i]
    return (fib*(hi-f["close"][i]))/(3.0*a)

SCREENS={
 "A": lambda f,i: f["dd252"][i]>=0.15 and _rr(f,i)>=1.5 and _oversold(f,i),
 "G": lambda f,i: (f["mom"][i] is not None and f["mom"][i]>0.10 and f["rsi"][i] is not None and f["rsi"][i]<=35),
 "E": lambda f,i: (f["ma200"][i] is not None and f["close"][i]>f["ma200"][i] and f["rsi"][i] is not None and f["rsi"][i]<=35),
 "H": lambda f,i: (f["mom"][i] is not None and f["mom"][i]<-0.10 and f["rsi"][i] is not None and f["rsi"][i]<=35),
}

MINPX=5.0; MINDV=1_000_000.0
def build(filt):
    sig={k:{} for k in SCREENS}
    for sym,f in F.items():
        if f["delisted"]: continue
        c=f["close"]; v=f["vol"]; n=len(c)
        # trailing 20d median dollar volume
        dv=[None]*n
        if v:
            for i in range(20,n):
                w=sorted(c[j]*v[j] for j in range(i-19,i+1))
                dv[i]=w[10]
        for i in range(252,n-1):
            if filt:
                if c[i]<MINPX: continue
                if dv[i] is None or dv[i]<MINDV: continue
            for k,t in SCREENS.items():
                try: hit=t(f,i)
                except Exception: hit=False
                if hit: sig[k].setdefault(f["dates"][i],[]).append((sym,i))
    return sig

for filt in (False,True):
    s=build(filt)
    pickle.dump(s, open(SP+f"sig_{'filt' if filt else 'raw'}.pkl","wb"),4)
    print("filter" if filt else "nofilter", {k:sum(len(v) for v in d.values()) for k,d in s.items()})
