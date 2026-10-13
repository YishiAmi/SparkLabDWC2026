from paths import *
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
SER=["gas_cum","oil_cum","water_cum","gas_rate","oil_rate","water_rate"]
D=np.load(DATA/"data.npz"); X=D["X"]
Yall=np.concatenate([D[k] for k in SER],1)   # 101 x 246
def feat(X):
    Z=np.array(X,float).copy(); Z[:,2]=np.log(Z[:,2]); Z[:,3]=np.log(Z[:,3]); return Z
class Proxy:
    def __init__(s,nc=8): s.nc=nc
    def fit(s,Xtr,Ytr):
        s.sx=StandardScaler().fit(feat(Xtr)); s.sy=StandardScaler().fit(Ytr)
        s.pca=PCA(s.nc).fit(s.sy.transform(Ytr)); S=s.pca.transform(s.sy.transform(Ytr)); Z=s.sx.transform(feat(Xtr))
        s.gps=[GPR(C(1.0)*RBF([1.0]*4,(1e-2,1e3))+WhiteKernel(1e-3,(1e-9,1e-1)),normalize_y=True,n_restarts_optimizer=4,random_state=k).fit(Z,S[:,k]) for k in range(s.nc)]
        return s
    def predict(s,Xn):
        Z=s.sx.transform(feat(Xn)); S=np.column_stack([g.predict(Z) for g in s.gps]); return s.sy.inverse_transform(s.pca.inverse_transform(S))
def split(P): return {k:P[:,i*41:(i+1)*41] for i,k in enumerate(SER)}
if __name__=="__main__":
    tr,va,te=slice(0,70),slice(70,85),slice(85,100)
    for nc in (6,8,10,12):
        m=Proxy(nc).fit(X[tr],Yall[tr]); out=[]
        for nm,s in (("val",va),("test",te)):
            P=split(m.predict(X[s])); T=split(Yall[s])
            out.append(nm+" "+" ".join(f"{k}:{np.mean(np.abs(P[k]-T[k]).sum(1)/np.abs(T[k]).sum(1))*100:.2f}%" for k in SER))
        print(nc,"|".join(out))
