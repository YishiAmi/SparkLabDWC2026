from paths import *
import numpy as np
from scipy.optimize import curve_fit
D=np.load(DATA/"data.npz"); T=1998+np.arange(41)/4.0
def forecast(qo,qg,qw,Np,Gp,Wp,ic,nf,W=12,mode="exp",wmode="wor"):
    """series arrays on quarterly grid (rate at node, cum at node); fit on nodes ic-W..ic, forecast nf quarters. returns cum arrays for nodes ic+1..ic+nf"""
    s=slice(ic-W,ic+1); t=T[s]-T[ic]; dt=0.25*365.25
    # oil rate
    y=np.log(qo[s])
    if mode=="exp":
        b1,b0=np.polyfit(t,y,1); q=lambda tt: np.exp(b0+b1*tt)
    else:
        f=lambda tt,q0,Dd,b: q0/(1+b*Dd*tt)**(1/b)
        p0=[qo[ic],0.2,0.5]
        try: p,_=curve_fit(f,t,qo[s],p0=p0,bounds=([0,1e-3,0.01],[1e5,5,1.5]),maxfev=20000)
        except Exception: p=p0
        q=lambda tt: f(tt,*p)
    tf=np.arange(1,nf+1)*0.25
    # cumulative via trapezoid on quarterly nodes using fine steps
    fine=np.arange(0,nf*0.25+1e-9,0.25/10); qf=q(fine)
    inc=np.concatenate([[0],np.cumsum((qf[1:]+qf[:-1])/2*(0.25/10)*365.25)])
    Nf=Np[ic]+inc[::10]; qo_f=q(np.concatenate([[0],tf]))
    Nf=Nf[1:]; qof=qo_f[1:]
    # gas: GOR linear in time
    gor=qg[s]/qo[s]; g1,g0=np.polyfit(t,gor,1); gorf=np.maximum(g0+g1*tf,0.0)
    # water: ln(WOR) vs Np linear (or vs time)
    wor=qw[s]/qo[s]
    if wmode=="wor":
        x=Np[s]; w1,w0=np.polyfit(x,np.log(wor),1); worf=np.exp(w0+w1*Nf)
    else:
        w1,w0=np.polyfit(t,np.log(wor),1); worf=np.exp(w0+w1*tf)
    qwf=worf*qof; qgf=gorf*qof
    # integrate gas, water rates by trapezoid on quarter nodes starting from node ic values
    def cumint(q0,qf_,C0):
        qq=np.concatenate([[q0],qf_]); return C0+np.cumsum((qq[1:]+qq[:-1])/2*0.25*365.25)
    return cumint(qo[ic],qof,Np[ic]),cumint(qg[ic],qgf,Gp[ic]),cumint(qw[ic],qwf,Wp[ic])
if __name__=="__main__":
    import itertools
    for ic,W,mode,wmode in itertools.product([28,32,36],[8,12,16],["exp","hyp"],["wor","time"]):
        errs=[]
        for j in range(100):
            o,g,w=forecast(D["oil_rate"][j],D["gas_rate"][j],D["water_rate"][j],D["oil_cum"][j],D["gas_cum"][j],D["water_cum"][j],ic,40-ic,W,mode,wmode)
            tr=[D["oil_cum"][j,-1],D["gas_cum"][j,-1],D["water_cum"][j,-1]]
            errs.append([(o[-1]-tr[0])/tr[0],(g[-1]-tr[1])/tr[1],(w[-1]-tr[2])/tr[2]])
        e=np.abs(np.array(errs)).mean(0)*100; b=np.array(errs).mean(0)*100
        print(f"cut {1998+ic/4:.2f} W{W//4}y {mode} {wmode:5s} MAPE oil/gas/water% {np.round(e,2)} bias {np.round(b,2)}")
