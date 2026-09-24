import re, json, numpy as np
log = open('cellH.log').read()
rows=[]; model=None
for line in log.splitlines():
    m=re.match(r'\s*\[\d/4\] (\S+) ', line)
    if m: model=m.group(1)
    m=re.match(r'\s*input (\d) block\s+(\d+) pos\s+(\d+): sigma1\s+([\d.]+) \(est\s+([\d.]+), (\d+) it\)\s+rho\s+([\d.]+) \(est\s+([\d.]+)\)\s+sigma1\(J-I\)\s+([\d.]+)\s+probes64\s+([\d.]+)\s+sigma1/rho\s+([\d.]+)', line)
    if m:
        i,b,p,s1,s1e,it,rho,rhoe,sb,pr,ratio=m.groups()
        rows.append(dict(model=model,inp=int(i),block=int(b),pos=int(p),sigma1=float(s1),sigma1_est=float(s1e),iters=int(it),
                         rho=float(rho),rho_est=float(rhoe),sigma1_branch=float(sb),probes64=float(pr),ratio=float(ratio)))
json.dump(rows, open('validation_cellH.json','w'), indent=1)
a=lambda k: np.array([r[k] for r in rows])
s1,s1e,rho,rhoe,sb,pr = a('sigma1'),a('sigma1_est'),a('rho'),a('rho_est'),a('sigma1_branch'),a('probes64')
print('n pairs', len(rows), '| models', sorted({r['model'] for r in rows}))
print('sigma1 estimator: max rel err %.2e, median %.2e, Pearson r %.6f' % (np.abs(s1e/s1-1).max(), np.median(np.abs(s1e/s1-1)), np.corrcoef(s1,s1e)[0,1]))
print('rho estimator:    max rel err %.2e, median %.2e' % (np.abs(rhoe/rho-1).max(), np.median(np.abs(rhoe/rho-1))))
print('random 64 probes: underestimate %.0f%%-%.0f%% (median %.0f%%)' % (100*(1-pr/s1).min(), 100*(1-pr/s1).max(), 100*np.median(1-pr/s1)))
print('sigma1/rho: %.1f-%.1f, median %.1f' % (a('ratio').min(), a('ratio').max(), np.median(a('ratio'))))
d=np.abs(sb-s1)
print('|sigma1(J-I)-sigma1(J)|: max %.3f, median %.3f, frac<=1 %.2f' % (d.max(), np.median(d), (d<=1).mean()))
print('   max at:', [(r['model'],r['block'],r['pos']) for r in rows if abs(r['sigma1_branch']-r['sigma1'])==d.max()])
print('iters: median %d, max %d' % (np.median(a('iters')), a('iters').max()))
for mo in sorted({r['model'] for r in rows}):
    sub=[r for r in rows if r['model']==mo]
    e=np.array([abs(r['sigma1_est']/r['sigma1']-1) for r in sub]); rr=np.array([r['ratio'] for r in sub])
    pp=np.array([1-r['probes64']/r['sigma1'] for r in sub]); dd=np.array([abs(r['sigma1_branch']-r['sigma1']) for r in sub])
    re_=np.array([abs(r['rho_est']/r['rho']-1) for r in sub])
    print(f"  {mo:26s} n={len(sub):3d} s1err {e.max():.1e}  rho_err {re_.max():.1e}  s1/rho {rr.min():.1f}-{rr.max():.1f}  probes -{100*pp.min():.0f}..-{100*pp.max():.0f}%  |dbranch| {dd.max():.3f}")
