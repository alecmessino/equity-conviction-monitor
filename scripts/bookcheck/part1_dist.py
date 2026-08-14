"""Distribution of event alpha per screen: mean vs median vs trimmed vs top-1% share."""
import pickle, statistics as st, sys, math
SP="/tmp/claude-0/-home-user-equity-conviction-monitor/58c3f77b-53ed-5f51-98fd-1c74f891c4af/scratchpad/"
D=pickle.load(open(SP+"feat.pkl","rb"))
F=D["feats"]; bidx=D["bidx"]; bc=D["bclose"]
spy_idx={d:i for i,d in enumerate(D["spy_dates"])}; spyc=D["spy_close"]
WARMUP=252

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
 "A baseline (dd>=15% + rr>=1.5 + oversold)": lambda f,i: f["dd252"][i]>=0.15 and _rr(f,i)>=1.5 and _oversold(f,i),
 "G momentum winner + dip":                   lambda f,i: (f["mom"][i] is not None and f["mom"][i]>0.10 and f["rsi"][i] is not None and f["rsi"][i]<=35),
 "E dip in uptrend":                          lambda f,i: (f["ma200"][i] is not None and f["close"][i]>f["ma200"][i] and f["rsi"][i] is not None and f["rsi"][i]<=35),
 "H momentum loser + dip":                    lambda f,i: (f["mom"][i] is not None and f["mom"][i]<-0.10 and f["rsi"][i] is not None and f["rsi"][i]<=35),
}

def run(hold, exclude_delisted=True):
    res={k:[] for k in SCREENS}
    for sym,f in F.items():
        if exclude_delisted and f["delisted"]: continue
        n=len(f["close"]); nxt={k:WARMUP for k in SCREENS}
        for i in range(WARMUP, n-hold):
            date=f["dates"][i]
            bi=bidx.get(date); si=spy_idx.get(date)
            if bi is None or bi+hold>=len(bc): continue
            beta=f["beta"][i]
            if beta is None: continue
            beta=max(0.2,min(3.0,beta))
            fwd=f["close"][i+hold]/f["close"][i]-1.0
            mkt=bc[bi+hold]/bc[bi]-1.0
            smkt=spyc[si+hold]/spyc[si]-1.0 if (si is not None and si+hold<len(spyc)) else None
            for name,test in SCREENS.items():
                if i<nxt[name]: continue
                try: hit=test(f,i)
                except Exception: hit=False
                if not hit: continue
                nxt[name]=i+hold
                res[name].append({"sym":sym,"date":date,"fwd":fwd,
                                  "ex":fwd-mkt,"al":fwd-beta*mkt,
                                  "spyex":(fwd-smkt) if smkt is not None else None})
    return res

def trimmed(v,p=0.05):
    v=sorted(v); k=int(len(v)*p)
    return st.mean(v[k:len(v)-k]) if len(v)-2*k>0 else float('nan')

for hold in (10,25):
    print(f"\n===== HOLD {hold} sessions | benchmark RSP (equal-weight S&P) | survivors only =====")
    res=run(hold)
    print(f"{'screen':<42}{'N':>7}{'mean':>8}{'med':>8}{'tr5%':>8}{'tr10%':>8}{'top1%share':>11}{'exTop1':>8}{'win%':>6}{'max':>9}{'worst':>8}")
    for name,ev in res.items():
        al=[e["al"] for e in ev]
        if len(al)<30: continue
        tot=sum(al); s=sorted(al,reverse=True); k=max(1,int(round(len(al)*0.01)))
        top=sum(s[:k])
        share=top/tot if tot!=0 else float('nan')
        ex_top=(tot-top)/(len(al)-k)
        mx=max(al); wr=min(al)
        print(f"{name:<42}{len(al):>7}{st.mean(al):>+8.2%}{st.median(al):>+8.2%}{trimmed(al,.05):>+8.2%}"
              f"{trimmed(al,.10):>+8.2%}{share:>10.0%}{ex_top:>+8.2%}{sum(1 for x in al if x>0)/len(al):>6.0%}{mx:>+9.1%}{wr:>+8.1%}")
    # raw excess too
    print(f"\n{'  (raw excess vs RSP, no beta adj)':<42}{'N':>7}{'mean':>8}{'med':>8}{'tr10%':>8}{'top1%share':>11}")
    for name,ev in res.items():
        ex=[e["ex"] for e in ev]
        if len(ex)<30: continue
        tot=sum(ex); s=sorted(ex,reverse=True); k=max(1,int(round(len(ex)*0.01)))
        print(f"{name:<42}{len(ex):>7}{st.mean(ex):>+8.2%}{st.median(ex):>+8.2%}{trimmed(ex,.10):>+8.2%}{(sum(s[:k])/tot if tot else float('nan')):>10.0%}")
    # top events
    for name in ("A baseline (dd>=15% + rr>=1.5 + oversold)","G momentum winner + dip"):
        ev=sorted(res[name], key=lambda e:-e["al"])[:8]
        print(f"  top events {name}: " + ", ".join(f"{e['sym']} {e['date']} {e['al']:+.0%}" for e in ev))
    # concentration: how many events to make half the total alpha
    for name,ev in res.items():
        al=sorted([e["al"] for e in ev],reverse=True)
        if len(al)<30: continue
        tot=sum(al); run_=0.0; c=0
        if tot>0:
            for x in al:
                run_+=x; c+=1
                if run_>=0.5*tot: break
            print(f"  {name}: top {c} of {len(al)} events ({c/len(al):.1%}) = 50% of total alpha")
