#!/usr/bin/env python3
"""
make_endlog_tw.py -- produce a CPS/jinv-style end.log TW block for ONE station,
computing the cross-convolution datum (data.y) and residual (data.res) with the
EXACT jinv2024 chain (readData.m window+taper+amax, syn_tw.m joint-L2, MyConv,
dpre.m residual).

Output matches end.log exactly:
    data5  nd = 200 rms =  R.RRR sd =  S.SSS  A.AAA
    TW  -5.00   0.00 res   0.00 sigma   0.02 on 1
    ... (200 lines)

Works for any synthetic (CPS syn.*, FDFK fdfk_*, etc). Also writes a matching
GMT-plottable file so `grep -w TW | awk '{print $2,$3}' | psxy` works.

USAGE:
  python3 make_endlog_tw.py \
      --syn_prefix syn.20150323045138_WB12 \
      --obs_prefix 20150323045138_WB12 \
      --syn_arr 0 --obs_arr 1070.35 \
      --sigma 0.02 --out end_WB12.log
"""
import numpy as np, argparse

MAXTW=20.0; TWPRE=5.0; DT=0.1
NN=int(round(MAXTW/DT)); TWPRE0=int(round(TWPRE/DT))+1; NTAP=int(round(0.5*TWPRE0))+1
def build_twtap():
    t=np.ones(NN); r=0.5*(1-np.cos(np.linspace(0,np.pi,NTAP))); t[:NTAP]=r; t[NN-NTAP:]=r[::-1]; return t
TWTAP=build_twtap()

def read_sac(f):
    raw=open(f,'rb').read();F=np.frombuffer(raw[:280],dtype='<f4');I=np.frombuffer(raw[280:440],dtype='<i4')
    return float(F[0]),float(F[5]),int(I[9]),np.frombuffer(raw[632:632+int(I[9])*4],dtype='<f4').astype(float)

def window_taper(fname,arr,flip=False):
    dt,b,n,tmp=read_sac(fname);n0=int(round((arr-TWPRE-b)/dt))
    idx=np.clip(np.arange(n0,n0+NN),0,n-1);seg=TWTAP*tmp[idx]
    return -seg if flip else seg

def MyConv(a,b):
    c=np.convolve(a,b);return TWTAP*c[TWPRE0-1:NN+TWPRE0-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--syn_prefix',required=True)
    ap.add_argument('--obs_prefix',required=True)
    ap.add_argument('--syn_arr',type=float,default=0.0)
    ap.add_argument('--obs_arr',type=float,default=1070.35)
    ap.add_argument('--sigma',type=float,default=0.02)
    ap.add_argument('--flip_z',action='store_true')
    ap.add_argument('--flip_r',action='store_true')
    ap.add_argument('--out',default='end_station.log')
    a=ap.parse_args()

    obsZ=window_taper(f"{a.obs_prefix}.z",a.obs_arr); obsR=window_taper(f"{a.obs_prefix}.r",a.obs_arr)
    amax=np.max(np.abs(np.stack([obsZ,obsR]))); obsZ/=amax; obsR/=amax
    synZ=window_taper(f"{a.syn_prefix}.z",a.syn_arr,flip=a.flip_z)
    synR=window_taper(f"{a.syn_prefix}.r",a.syn_arr,flip=a.flip_r)
    sd_l2=np.sqrt(np.sum(synZ**2)+np.sum(synR**2)); Zhat=synZ/sd_l2; Rhat=synR/sd_l2

    data_y=MyConv(Rhat,obsZ)       # $3
    data_yp=MyConv(Zhat,obsR)
    res=data_y-data_yp             # $5
    tt=-TWPRE+np.arange(NN)*DT

    # end.log-style stats: rms of residual; sd values (like data5 line)
    rms=np.sqrt(np.mean(res**2))
    sd_obs=a.sigma/amax if amax>0 else a.sigma        # per-sample noise (as data.sd)
    sd2=np.std(res)
    chi=rms/a.sigma

    with open(a.out,'w') as f:
        f.write(f"data5  nd = {NN} rms = {rms:6.3f} sd = {sd_obs:6.3f} {sd2:6.3f}\n")
        for k in range(NN):
            f.write(f"TW {tt[k]:6.2f} {data_y[k]:6.2f} res {res[k]:6.2f} "
                    f"sigma {a.sigma:6.2f} on 1\n")
    print(f"wrote {a.out}  ({NN} TW lines)")
    print(f"  rms={rms:.3f}  chi={chi:.1f}")
    print(f"  GMT: grep -w TW {a.out} | awk '{{print $2,$3}}' | psxy   (data.y, black)")
    print(f"       grep -w TW {a.out} | awk '{{print $2,$3-$5}}' | psxy (fit, red)")

if __name__=='__main__': main()
