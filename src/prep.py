from paths import *
import pickle, numpy as np
P=pickle.load(open(DATA/"prod.pkl","rb"))
U=pickle.load(open(DATA/"unc.pkl","rb"))
X=np.array([r[1:] for r in U[1:]],float)  # 100x4
qd=np.array([f"{y}-{m:02d}-01" for y in range(1998,2008) for m in (1,4,7,10)]+["2008-01-01"],dtype="datetime64[D]")
names=["gas_cum","gas_rate","oil_cum","oil_rate","water_cum","water_rate"]
keys=["Gas production cumulative","Gas production rate","Oil production cumulative","Oil production rate","Water production cumulative","Water production rate"]
Y={}
for n,k in zip(names,keys):
    d,h,a=P[k]; d=np.array(d,dtype="datetime64[D]").astype(int).astype(float)
    out=np.full((101,len(qd)),np.nan)
    for j in range(101):
        m=~np.isnan(a[:,j])
        t=d[m]; v=a[m,j]
        # average duplicates not needed; interpolate linearly
        out[j]=np.interp(qd.astype(int).astype(float),t,v)
    Y[n]=out
np.savez(DATA/"data.npz",X=X,qd=qd.astype(int),**Y)
print({k:v.shape for k,v in Y.items()})
print("hist final oil/gas/water",[Y[n][100,-1] for n in ("oil_cum","gas_cum","water_cum")])
print("case range final oil",Y["oil_cum"][:100,-1].min(),Y["oil_cum"][:100,-1].max())
print("X ranges",X.min(0),X.max(0))
