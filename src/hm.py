from paths import *
import numpy as np, pickle
from proxy import *
from scipy.optimize import minimize
H=Yall[100]
sc=np.array([np.abs(Yall[:100,i*41:(i+1)*41]).max() for i in range(6)])
wts=np.array([1,1,1,.5,.5,.5])
def misfit(P):
    out=0
    for i in range(6):
        a=slice(i*41+(6 if i>=3 else 0),(i+1)*41)
        out=out+wts[i]*np.mean(((P[:,a]-H[a])/sc[i])**2,1)
    return np.sqrt(out/wts.sum())
if __name__=="__main__":
    m=Proxy(10).fit(X[:100],Yall[:100]); pickle.dump(m,open(MODELS/"proxy_final.pkl","wb"))
    lo=X[:100].min(0); hi=X[:100].max(0); rng=np.random.default_rng(0); N=100000
    S=lo+(hi-lo)*rng.random((N,4))
    mf=np.concatenate([misfit(m.predict(S[i:i+5000])) for i in range(0,N,5000)])
    mfc=misfit(Yall[:100])
    print("real cases best:",[(int(i)+1,round(float(mfc[i]),5)) for i in np.argsort(mfc)[:5]], " worst",mfc.max().round(4),"median",np.median(mfc).round(4))
    print("proxy-sample best",mf.min().round(5),"median",np.median(mf).round(4))
    # polish top 20 with Nelder-Mead
    nrm=lambda p:(p-lo)/(hi-lo)
    f=lambda z:(misfit(m.predict((lo+(hi-lo)*np.clip(z,0,1))[None]))[0])
    best=[]
    for i in np.argsort(mf)[:20]:
        r=minimize(f,nrm(S[i]),method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-8,maxiter=300))
        best.append((r.fun,lo+(hi-lo)*np.clip(r.x,0,1)))
    best.sort(key=lambda t:t[0]); print("polished best",[round(b[0],5) for b in best[:5]]); print(best[0][1])
    pickle.dump((S,mf,mfc,lo,hi,best),open(OUT/"hm_samples.pkl","wb"))
    # posterior-like set: accept within tolerance
    tol=np.percentile(mf,0.1); acc=S[mf<=tol]; print("accepted",len(acc),"tol",tol.round(5)); print("min",acc.min(0).round(3),"max",acc.max(0).round(3))
    print("corr of params w/ misfit (spearman-ish):",[np.round(np.corrcoef(S[:,k],mf)[0,1],2) for k in range(4)])
