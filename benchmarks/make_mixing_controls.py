"""Fixed-seed paired DGP ladder for diagnosing, NOT replacing, the chapter.

All variants have two continuous outcomes, one binary outcome and sign-changing
effects. Each one changes ONE difficulty relative to baseline, with identical
covariates, treatment assignment and underlying random numbers. These are
model-compatible simulations, not draws from the complete model prior or SBC.
Uses the chapter benchmark archive schema; continuous truths are log effects.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr


def generate(seed=20260907, n=240, periods=12):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n,2))
    sign = np.where(x[:,0]>=0,1.,-1.)
    z = np.zeros((n,periods))
    z[rng.permutation(n)[:n//2],4:] = 1
    ze = np.zeros_like(z); ze[:,4:] = 1
    s = np.maximum(np.arange(periods)-3,0)
    curve = 1-np.exp(-s/3)
    u,v,w = rng.normal(size=(3,n,periods))
    intercept = rng.normal(0,.3,(3,n,1))
    for variant in ('baseline','rare','correlated','unit_intercepts','complex_shape'):
        correlated = variant == 'correlated'
        ri = intercept if variant == 'unit_intercepts' else np.zeros_like(intercept)
        base = .20*x[:,1,None]+.025*np.arange(periods)
        mu_q = (-1.6 if variant == 'rare' else -.3)+base+ri[2]
        tau_g = .35*sign[:,None]*curve
        tau_h = -.30*sign[:,None]*curve
        tau_q = -.55*sign[:,None]*curve
        if variant == 'complex_shape':
            # Group-specific growth/fade, retaining benefit/harm at the target.
            shape = np.where(sign[:,None]>0,curve, .6*curve+.4*s/2*np.exp(1-s/2))
            tau_g = .35*sign[:,None]*shape
        e_h = .6*u+.8*v if correlated else v
        e_q = .45*u+.10*v+np.sqrt(.7875)*w if correlated else w
        yield variant, dict(x=x.astype('float32'),z=z.astype('float32'),ze=ze.astype('float32'),
            t=np.arange(1,periods+1),col=np.array(periods-1),
            gmv=base+ri[0]+tau_g*z+.3*u,
            hours=base+ri[1]+tau_h*z+.3*e_h,
            complaint=(mu_q+tau_q*z+e_q>0).astype(float),
            truth_gmv=tau_g[:,-1],truth_hours=tau_h[:,-1],
            truth_complaint=ndtr(mu_q[:,-1]+tau_q[:,-1])-ndtr(mu_q[:,-1]),
            listings=np.where(sign>0,100,20),fulfilled=(sign>0).astype(int))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260907)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in generate(args.seed):
        path = args.output/f'{name}-{args.seed}.npz'
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path,**data)
        print(name,'complaint prevalence',data['complaint'].mean(),flush=True)
