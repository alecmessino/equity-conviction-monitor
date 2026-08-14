"""Simulate a finite-slot book. Daily mark-to-market. Costs. vs RSP."""
import pickle, math, random, statistics as st, argparse, sys
SP="/tmp/claude-0/-home-user-equity-conviction-monitor/58c3f77b-53ed-5f51-98fd-1c74f891c4af/scratchpad/"
D=pickle.load(open(SP+"feat.pkl","rb"))
F=D["feats"]; BD=D["bdates"]; BC=D["bclose"]; BIDX=D["bidx"]

ap=argparse.ArgumentParser()
ap.add_argument("--screen",default="A")
ap.add_argument("--hold",type=int,default=25)
ap.add_argument("--slots",type=int,default=10)
ap.add_argument("--cost",type=float,default=20.0)   # bp round trip
ap.add_argument("--seeds",type=int,default=100)
ap.add_argument("--filt",type=int,default=1)
ap.add_argument("--pick",default="random")          # random|rsi|alpha
ap.add_argument("--entry",default="nextopen")       # nextopen|close
ap.add_argument("--cash",default="zero")            # zero|bench
ap.add_argument("--start",default="2014-09-02")
ap.add_argument("--json",default="")
a=ap.parse_args()

SIG=pickle.load(open(SP+f"sig_{'filt' if a.filt else 'raw'}.pkl","rb"))[a.screen]
dates=[d for d in BD if d>=a.start]
di={d:k for k,d in enumerate(dates)}
bpx=[BC[BIDX[d]] for d in dates]
cost=a.cost/10000.0

def px(sym,j,kind):
    f=F[sym]
    return f["open"][j] if kind=="open" else f["close"][j]

def simulate(seed):
    rng=random.Random(seed)
    eq=1.0; cash=1.0; pos=[]   # each: sym, shares_value_frac, entry_idx_in_name, exit_date_k
    equity=[]; trades=[]; wanted=0; taken=0; exposure=[]
    for k,d in enumerate(dates):
        # 1. exits at today's open (positions whose exit day is today)
        keep=[]
        for p in pos:
            if p["exit_k"]==k:
                f=F[p["sym"]]; j=p["exit_j"]
                v=p["val"]*(px(p["sym"],j,p["exit_kind"])/p["entry_px"])
                v*= (1.0-cost)
                cash+=v
                trades.append((v/p["val"]-1.0, p["sym"], d, v-p["val"]))
            else: keep.append(p)
        pos=keep
        # 2. entries: signals from yesterday's close (entry at today's open) or today's close
        if a.entry=="nextopen":
            sd = dates[k-1] if k>0 else None
            ekind="open"; ej_off=0
        else:
            sd = d; ekind="close"; ej_off=0
        cands=[]
        if sd:
            held={p["sym"] for p in pos}
            for sym,i in SIG.get(sd,[]):
                if sym in held: continue
                f=F[sym]
                j=f["idx"].get(d)
                if j is None: continue
                # need the exit bar to exist
                if j+a.hold>=len(f["close"]): continue
                ed=f["dates"][j+a.hold]
                if ed not in di: continue
                cands.append((sym,i,j,ed))
        wanted+=len(cands)
        free=a.slots-len(pos)
        if free>0 and cands:
            if a.pick=="random": rng.shuffle(cands)
            elif a.pick=="rsi": cands.sort(key=lambda c: F[c[0]]["rsi"][c[1]] if F[c[0]]["rsi"][c[1]] is not None else 99)
            else: cands.sort()
            # equal weight on current NAV
            nav=cash+sum(p["val"]*(F[p["sym"]]["close"][F[p["sym"]]["idx"].get(d, p["exit_j"])]/p["entry_px"]) for p in pos)
            for sym,i,j,ed in cands[:free]:
                w=nav/a.slots
                w=min(w,cash)
                if w<=1e-9: break
                cash-=w
                pos.append({"sym":sym,"val":w*(1.0),"entry_px":px(sym,j,ekind),
                            "exit_k":di[ed],"exit_j":F[sym]["idx"][ed],"exit_kind":ekind})
                taken+=1
        # 3. mark to market at close
        mv=0.0
        for p in pos:
            f=F[p["sym"]]; j=f["idx"].get(d)
            if j is None: j=p["exit_j"]
            mv+=p["val"]*(f["close"][j]/p["entry_px"])
        nav=cash+mv
        if a.cash=="bench" and k>0 and cash>0:
            cash*= bpx[k]/bpx[k-1]
            nav=cash+mv
        equity.append(nav)
        exposure.append(mv/nav if nav>0 else 0)
    return equity,trades,wanted,taken,exposure

def stats(eq,bench):
    rets=[eq[i]/eq[i-1]-1 for i in range(1,len(eq))]
    brets=[bench[i]/bench[i-1]-1 for i in range(1,len(bench))]
    yrs=len(eq)/252.0
    cagr=(eq[-1]/eq[0])**(1/yrs)-1
    bcagr=(bench[-1]/bench[0])**(1/yrs)-1
    sd=st.pstdev(rets); bsd=st.pstdev(brets)
    sh=(st.mean(rets)/sd*math.sqrt(252)) if sd>0 else 0
    bsh=(st.mean(brets)/bsd*math.sqrt(252)) if bsd>0 else 0
    def mdd(x):
        pk=x[0]; m=0.0
        for v in x:
            pk=max(pk,v); m=min(m,v/pk-1)
        return m
    # daily regression of book on benchmark
    n=len(rets); mx=st.mean(brets); my=st.mean(rets)
    cov=sum((brets[i]-mx)*(rets[i]-my) for i in range(n))/n
    var=st.pvariance(brets)
    beta=cov/var if var>0 else 0.0
    alpha_d=my-beta*mx
    resid=[rets[i]-alpha_d-beta*brets[i] for i in range(n)]
    se=(st.pstdev(resid)/math.sqrt(n)) if n>1 else 1.0
    tstat=alpha_d/se if se>0 else 0.0
    # vol-matched benchmark: lever RSP to the book's vol (no financing charge = generous)
    lev=(sd/bsd) if bsd>0 else 1.0
    v=1.0
    for r in brets: v*= (1+lev*r)
    volmatch=v**(1/yrs)-1
    return dict(beta=beta, alpha_ann=alpha_d*252, tstat=tstat, lev=lev, volmatch=volmatch,
                cagr=cagr,bcagr=bcagr,sharpe=sh,bsharpe=bsh,vol=sd*math.sqrt(252),
                bvol=bsd*math.sqrt(252),mdd=mdd(eq),bmdd=mdd(bench),total=eq[-1]/eq[0]-1,
                btotal=bench[-1]/bench[0]-1)

out=[]
for s in range(a.seeds):
    eq,tr,w,t,ex=simulate(1000+s)
    r=stats(eq,bpx); r["trades"]=len(tr); r["tr"]=tr; r["wanted"]=w; r["taken"]=t
    r["expo"]=sum(ex)/len(ex); r["eq"]=eq
    out.append(r)

def q(key,p):
    v=sorted(o[key] for o in out); return v[min(len(v)-1,int(p*len(v)))]

r0=out[0]
print(f"screen {a.screen} hold {a.hold} slots {a.slots} cost {a.cost:.0f}bp entry {a.entry} "
      f"pick {a.pick} cash {a.cash} filt {a.filt} seeds {a.seeds}")
print(f"  window {dates[0]} .. {dates[-1]}  ({len(dates)/252:.1f}y)")
print(f"  signals wanted {r0['wanted']}  taken {q('taken',.5)}  fill {q('taken',.5)/max(r0['wanted'],1):.2%}"
      f"  avg exposure {q('expo',.5):.0%}")
print(f"  BOOK   CAGR med {q('cagr',.5):+.2%}  [p5 {q('cagr',.05):+.2%}, p95 {q('cagr',.95):+.2%}]"
      f"  Sharpe {q('sharpe',.5):.2f}  maxDD {q('mdd',.5):.0%}  vol {q('vol',.5):.0%}")
print(f"  RSP    CAGR     {r0['bcagr']:+.2%}                        Sharpe {r0['bsharpe']:.2f}"
      f"  maxDD {r0['bmdd']:.0%}  vol {r0['bvol']:.0%}")
print(f"  book beta vs RSP {q('beta',.5):.2f}   ann. alpha {q('alpha_ann',.5):+.2%}  t={q('tstat',.5):.2f}"
      f"   vol-matched RSP ({q('lev',.5):.2f}x) CAGR {q('volmatch',.5):+.2%}")
print(f"  excess CAGR med {q('cagr',.5)-r0['bcagr']:+.2%}   P(book beats RSP) = "
      f"{sum(1 for o in out if o['cagr']>o['bcagr'])/len(out):.0%}")
# trade attribution on the median seed
mid=sorted(range(len(out)),key=lambda k:out[k]['cagr'])[len(out)//2]
tr=out[mid]["tr"]
rs=sorted(x[0] for x in tr)
import statistics as _st
print(f"  median-seed trades n={len(tr)} mean {_st.mean(rs):+.2%} median {_st.median(rs):+.2%} "
      f"best {max(rs):+.1%} worst {min(rs):+.1%} win {sum(1 for x in rs if x>0)/len(rs):.0%}")
top=sorted(tr,key=lambda x:-x[3])[:5]
print("  top P&L trades: "+", ".join(f"{t[1]} {t[2]} {t[0]:+.0%}" for t in top))
if a.json:
    import json
    json.dump({"cfg":vars(a),"med_cagr":q('cagr',.5),"p5":q('cagr',.05),"p95":q('cagr',.95),
               "bcagr":r0['bcagr'],"sharpe":q('sharpe',.5),"mdd":q('mdd',.5),
               "bsharpe":r0['bsharpe'],"bmdd":r0['bmdd'],"fill":q('taken',.5)/max(r0['wanted'],1),
               "expo":q('expo',.5),"pbeat":sum(1 for o in out if o['cagr']>o['bcagr'])/len(out),
               "trades":q('trades',.5),"eq_med":out[sorted(range(len(out)),key=lambda k:out[k]['cagr'])[len(out)//2]]["eq"],
               "dates":dates},
              open(a.json,"w"))
