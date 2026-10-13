from paths import *
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=np.load(DATA/"data.npz"); R=np.load(OUT/"fc20.npz")["res"]; t=1998+np.arange(41)/4; tf=2008+np.arange(1,81)/4
fig,ax=plt.subplots(1,3,figsize=(16,4.5))
for c,(k,i,n) in enumerate([("oil_cum",0,"Oil cumulative [STB]"),("gas_cum",1,"Gas cumulative [MSCF]"),("water_cum",2,"Water cumulative [STB]")]):
    for j in range(100): ax[c].plot(t,D[k][j],color="0.85",lw=.6)
    ax[c].plot(t,D[k][100],"k.",ms=4,label="Observed history")
    for j in range(10): ax[c].plot(tf,R[j,i],lw=1.2,color="tab:red",alpha=.7)
    ax[c].plot([],[],color="tab:red",label="10 matched forecasts"); ax[c].axvline(2008,ls="--",color="k"); ax[c].axvline(2018,ls=":",color="gray"); ax[c].set_title(n); ax[c].legend()
plt.tight_layout(); plt.savefig(OUT/"forecast_overview.png",dpi=80)
