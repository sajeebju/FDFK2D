#!/usr/bin/env python3
"""
tw_2d_crossconv.py -- from FDFK2D 2D seismograms (seisx/seisz.su), extract a
station, post-process to the jinv TW window, and plot the cross-convolution
residual against observed, using the EXACT jinv MyConv/normalization chain.

Steps:
  1. Read 2D seisx.su/seisz.su (many receivers).
  2. Pick the receiver matching the station's x-position along the array.
  3. Auto-detect the S arrival (~30 s), decimate to 0.1 s, window -5..+15 s,
     apply polarity flips -> synthetic Z/R on the jinv grid.
  4. Window+taper obs at its S-arrival, normalize (amax).
  5. jinv chain: Zhat/Rhat (joint L2), data.y=MyConv(Rhat,obsZ),
     data.yp=MyConv(Zhat,obsR), residual=y-yp.
  6. Plot black=data.y, red=fit(=y-res), + residual.

USAGE:
  python3 tw_2d_crossconv.py \
      --seis_dir /home/yaoj/C++/FDFK2D/input_2d/seismograms \
      --obs_prefix /home/yaoj/C++/grid.jinv2024/grid-1/20150323045138_WB12 \
      --station_x_km 60 --model_x0_km -171 --model_x1_km 171 \
      --obs_arr 1070.35 --flip_z --flip_r
"""
import numpy as np, argparse
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# jinv globals
MAXTW=20.0; TWPRE=5.0; DT=0.1
NN=int(round(MAXTW/DT)); TWPRE0=int(round(TWPRE/DT))+1; NTAP=int(round(0.5*TWPRE0))+1
def build_twtap():
    tap=np.ones(NN); r=0.5*(1-np.cos(np.linspace(0,np.pi,NTAP)))
    tap[:NTAP]=r; tap[NN-NTAP:]=r[::-1]; return tap
TWTAP=build_twtap()

def read_su(f):
    raw=open(f,'rb').read(); h=np.frombuffer(raw[:240],dtype='<i2'); ns=int(h[57])
    tb=240+ns*4; ntr=len(raw)//tb; d=np.zeros((ns,ntr),dtype=np.float32)
    for i in range(ntr):
        o=i*tb+240; d[:,i]=np.frombuffer(raw[o:o+ns*4],dtype='<f4')
    return d, ns, ntr

def read_sac(f):
    raw=open(f,'rb').read(); F=np.frombuffer(raw[:280],dtype='<f4'); I=np.frombuffer(raw[280:440],dtype='<i4')
    return float(F[0]),float(F[5]),int(I[9]),np.frombuffer(raw[632:632+int(I[9])*4],dtype='<f4').astype(float)

def decimate(tr, dt_in, dt_out=DT):
    fac=int(round(dt_out/dt_in)); b,a=butter(6,(0.5/dt_out)/(0.5/dt_in),btype='low')
    return filtfilt(b,a,tr)[::fac]

def detect_peak(zr, dt, search_after=2.0):
    e=np.abs(zr).copy(); e[:int(search_after/dt)]=0; return int(np.argmax(e))

def MyConv(a,b):
    c=np.convolve(a,b); return TWTAP*c[TWPRE0-1:NN+TWPRE0-1]

def window_taper_sac(fname, arr):
    dt,b,n,tmp=read_sac(fname); n0=int(round((arr-TWPRE-b)/dt))
    idx=np.clip(np.arange(n0,n0+NN),0,n-1); return TWTAP*tmp[idx]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seis_dir',required=True,help='dir with seisx.su/seisz.su')
    ap.add_argument('--obs_prefix',required=True)
    ap.add_argument('--station_x_km',type=float,required=True,help='station x along array (km)')
    ap.add_argument('--model_x0_km',type=float,required=True,help='model left edge x (km)')
    ap.add_argument('--model_x1_km',type=float,required=True,help='model right edge x (km)')
    ap.add_argument('--obs_arr',type=float,default=1070.35)
    ap.add_argument('--flip_z',action='store_true')
    ap.add_argument('--flip_r',action='store_true')
    ap.add_argument('--sigma',type=float,default=0.02)
    ap.add_argument('--out',default='tw_2d_crossconv.png')
    a=ap.parse_args()

    X=read_su(f"{a.seis_dir}/seisx.su"); Z=read_su(f"{a.seis_dir}/seisz.su")
    (Xd,nsx,ntr)=X; (Zd,_,_)=Z
    dt_su=0.01
    # receiver index for the station x-position
    frac=(a.station_x_km-a.model_x0_km)/(a.model_x1_km-a.model_x0_km)
    ri=int(round(frac*(ntr-1))); ri=max(0,min(ri,ntr-1))
    print(f"station x={a.station_x_km} km -> receiver index {ri}/{ntr}")

    x=Xd[:,ri].astype(float); z=Zd[:,ri].astype(float)
    # decimate, detect S peak, window -5..+15
    xr=decimate(x,dt_su); zr=decimate(z,dt_su)
    i_arr=detect_peak(zr,DT); print(f"S arrival at t={i_arr*DT:.1f}s in trace")
    pre=int(round(TWPRE/DT)); start=i_arr-pre
    def cut(v):
        out=np.zeros(NN); s=max(start,0); e=min(start+NN,len(v)); out[s-start:s-start+(e-s)]=v[s:e]; return out
    sgnz=-1 if a.flip_z else 1; sgnr=-1 if a.flip_r else 1
    synZ=TWTAP*(sgnz*cut(zr)); synR=TWTAP*(sgnr*cut(xr))

    # obs
    obsZ=window_taper_sac(f"{a.obs_prefix}.z",a.obs_arr)
    obsR=window_taper_sac(f"{a.obs_prefix}.r",a.obs_arr)
    amax=np.max(np.abs(np.stack([obsZ,obsR]))); obsZ/=amax; obsR/=amax

    # jinv chain
    sd=np.sqrt(np.sum(synZ**2)+np.sum(synR**2)); Zhat=synZ/sd; Rhat=synR/sd
    data_y=MyConv(Rhat,obsZ); data_yp=MyConv(Zhat,obsR); res=data_y-data_yp
    fit=data_y-res
    tt=-TWPRE+np.arange(NN)*DT
    cc=np.corrcoef(data_y,fit)[0,1]; rms=np.sqrt(np.mean(res**2)); chi=rms/a.sigma

    fig,ax=plt.subplots(2,1,figsize=(11,7))
    ax[0].plot(tt,data_y,'k',lw=1.6,label='data.y = MyConv(Rhat,obsZ)  ($3)')
    ax[0].plot(tt,fit,'r',lw=1.3,label='fit = data.y-res  ($3-$5)')
    ax[0].axvline(0,color='gray',ls=':'); ax[0].set_xlim(-5,15)
    ax[0].set_title(f"2D FDFK TW cross-conv  CC={cc:.3f}  res_rms={rms:.3f}  chi={chi:.1f}")
    ax[0].set_xlabel('time (s)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(tt,res,'b',lw=1.1); ax[1].axvline(0,color='gray',ls=':'); ax[1].set_xlim(-5,15)
    ax[1].set_title('residual ($5)'); ax[1].set_xlabel('time (s)'); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(a.out,dpi=120,bbox_inches='tight')
    print(f"CC={cc:.3f} res_rms={rms:.3f} chi={chi:.1f} -> {a.out}")

if __name__=='__main__': main()
