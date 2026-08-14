"""Book-level concentration + per-year returns for the median seed."""
import pickle, math, random, statistics as st, collections
SP="/tmp/claude-0/-home-user-equity-conviction-monitor/58c3f77b-53ed-5f51-98fd-1c74f891c4af/scratchpad/"
D=pickle.load(open(SP+"feat.pkl","rb")); F=D["feats"]; BD=D["bdates"]; BC=D["bclose"]; BIDX=D["bidx"]
START="2014-09-02"; COST=0.0020
dates=[d for d in BD if d>=START]; di={d:k for k,d in enumerate(dates)}
bpx=[BC[BIDX[d]] for d in dates]

def sim(SIG,hold,slots,seed):
    rng=random.Random(seed); cash=1.0; pos=[]; equity=[]; tr=[]
    for k,d in enumerate(dates):
        keep=[]
        for p in pos:
            if p["exit_k"]==k:
                f=F[p["sym"]]; v=p["val"]*(f["open"][p["exit_j"]]/p["entry_px"])*(1-COST)
                cash+=v; tr.append((v-p["val"], v/p["val"]-1, p["sym"], p["d"]))
            else: keep.append(p)
        pos=keep
        sd=dates[k-1] if k>0 else None
        cands=[]
        if sd:
            held={p["sym"] for p in pos}
            for sym,i in SIG.get(sd,[]):
                if sym in held: continue
                f=F[sym]; j=f["idx"].get(d)
                if j is None or j+hold>=len(f["close"]): continue
                ed=f["dates"][j+hold]
                if ed in di: cands.append((sym,j,ed))
        free=slots-len(pos)
        if free>0 and cands:
            rng.shuffle(cands)
            nav=cash+sum(p["val"]*(F[p["sym"]]["close"][F[p["sym"]]["idx"].get(d,p["exit_j"])]/p["entry_px"]) for p in pos)
            for sym,j,ed in cands[:free]:
                w=min(nav/slots,cash)
                if w<=1e-9: break
                cash-=w
                pos.append({"sym":sym,"val":w,"entry_px":F[sym]["open"][j],"exit_k":di[ed],
                            "exit_j":F[sym]["idx"][ed],"d":d})
        mv=sum(p["val"]*(F[p["sym"]]["close"][F[p["sym"]]["idx"].get(d,p["exit_j"])]/p["entry_px"]) for p in pos)
        equity.append(cash+mv)
    return equity,tr

for scr in ("A","G"):
    SIG=pickle.load(open(SP+"sig_filt.pkl","rb"))[scr]
    for slots in (5,10):
        runs=[]
        for s in range(31):
            eq,tr=sim(SIG,25,slots,1000+s); runs.append((eq[-1],eq,tr))
        runs.sort(key=lambda x:x[0]); _,eq,tr=runs[len(runs)//2]
        # per-year
        yr=collections.defaultdict(lambda:[None,None]); byr=collections.defaultdict(lambda:[None,None])
        for k,d in enumerate(dates):
            y=d[:4]
            if yr[y][0] is None: yr[y][0]=eq[k-1] if k>0 else eq[0]; byr[y][0]=bpx[k-1] if k>0 else bpx[0]
            yr[y][1]=eq[k]; byr[y][1]=bpx[k]
        print(f"\n== screen {scr}, 25-session hold, {slots} slots, 20bp, median-of-31 seed ==")
        print("  year   book    RSP  excess")
        tot=0
        for y in sorted(yr):
            b=yr[y][1]/yr[y][0]-1; m=byr[y][1]/byr[y][0]-1
            print(f"  {y}  {b:+7.1%}{m:+7.1%}{b-m:+8.1%}")
        # concentration of P&L
        pl=sorted([t[0] for t in tr],reverse=True)
        tp=sum(pl)
        k5=max(1,int(round(len(pl)*0.01)))
        print(f"  trades {len(tr)}  total \$P&L on \$1 book {tp:+.3f}   top 1% ({k5} trades) = {sum(pl[:k5])/tp:.0%} of P&L"
              f"   top 10 trades = {sum(pl[:10])/tp:.0%}")
        print("  biggest: "+", ".join(f"{t[2]} {t[3]} {t[1]:+.0%}" for t in sorted(tr,key=lambda x:-x[0])[:6]))
