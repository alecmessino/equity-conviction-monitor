"""Event distribution with and without the tradeability filter (px>=$5, ADV>=$1M)."""
import pickle, statistics as st
SP="/tmp/claude-0/-home-user-equity-conviction-monitor/58c3f77b-53ed-5f51-98fd-1c74f891c4af/scratchpad/"
D=pickle.load(open(SP+"feat.pkl","rb")); F=D["feats"]; BIDX=D["bidx"]; BC=D["bclose"]

def trimmed(v,p):
    v=sorted(v); k=int(len(v)*p)
    return st.mean(v[k:len(v)-k])

for tag in ("raw","filt"):
    SIG=pickle.load(open(SP+f"sig_{tag}.pkl","rb"))
    for hold in (10,25):
        print(f"\n--- {tag} filter, hold {hold} ---")
        print(f"{'scr':<4}{'N':>7}{'mean':>8}{'med':>8}{'tr10%':>8}{'top1%':>7}{'exTop1':>8}{'win%':>6}{'n50%':>7}")
        for scr,byd in SIG.items():
            # rebuild per-symbol event list with one-open-position-per-name throttle
            ev=[]
            per={}
            for d,lst in byd.items():
                for sym,i in lst: per.setdefault(sym,[]).append(i)
            for sym,idxs in per.items():
                f=F[sym]; nxt=-1
                for i in sorted(idxs):
                    if i<nxt: continue
                    if i+hold>=len(f["close"]): break
                    bi=BIDX.get(f["dates"][i])
                    if bi is None or bi+hold>=len(BC): continue
                    b=f["beta"][i]
                    if b is None: continue
                    b=max(0.2,min(3.0,b))
                    fwd=f["close"][i+hold]/f["close"][i]-1.0
                    mkt=BC[bi+hold]/BC[bi]-1.0
                    ev.append((fwd-b*mkt, sym, f["dates"][i]))
                    nxt=i+hold
            al=[e[0] for e in ev]
            if len(al)<30: continue
            tot=sum(al); s=sorted(al,reverse=True); k=max(1,int(round(len(al)*0.01)))
            r=0.0; c=0
            for x in s:
                r+=x; c+=1
                if r>=0.5*tot: break
            print(f"{scr:<4}{len(al):>7}{st.mean(al):>+8.2%}{st.median(al):>+8.2%}{trimmed(al,.10):>+8.2%}"
                  f"{sum(s[:k])/tot:>7.0%}{(tot-sum(s[:k]))/(len(al)-k):>+8.2%}"
                  f"{sum(1 for x in al if x>0)/len(al):>6.0%}{c:>7}")
            if tag=="filt" and hold==25 and scr in("A","G"):
                print("     top: "+", ".join(f"{e[1]} {e[2]} {e[0]:+.0%}" for e in sorted(ev,key=lambda x:-x[0])[:6]))
