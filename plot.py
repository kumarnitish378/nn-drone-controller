import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]; INK="#0b0b0b"; SEC="#52514e"; BG="#fcfcfb"
plt.rcParams.update({"font.size":10,"axes.edgecolor":"#c9c8c2","axes.labelcolor":SEC,"xtick.color":SEC,
 "ytick.color":SEC,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e8e7e2",
 "grid.linewidth":0.6,"figure.facecolor":BG,"axes.facecolor":BG,"axes.titlesize":11,"axes.titleweight":"bold",
 "axes.titlelocation":"left","axes.titlecolor":INK})
d=np.load("flight_log.npz"); log=json.load(open(sys.argv[2] if len(sys.argv)>2 else "quad_log.json")); t=d["t"]; p=d["p"]; M=d["mission"]
fig,ax=plt.subplots(5,1,figsize=(7,17),gridspec_kw=dict(height_ratios=[1.5,1,1,1,1]))
a=ax[0]; a.plot(M[:,0],M[:,1],"--",color=SEC,lw=1,label="Planned path"); a.plot(p[:,0],p[:,1],color=C[0],lw=2,label="Flown path")
a.scatter(M[:,0],M[:,1],s=60,color=BG,edgecolor=SEC,zorder=3)
for i,m in enumerate(M[:-1]): a.annotate(f"WP{i+1}",(m[0],m[1]),textcoords="offset points",xytext=(8,8),color=SEC)
a.set_aspect("equal"); a.set_xlabel("x forward (m)"); a.set_ylabel("y left (m)"); a.set_title("Waypoint mission, top view"); a.legend(frameon=False,loc="center")
a=ax[1]; a.plot(t,d["tgt"][:,2],"--",color=SEC,lw=1,label="Target"); a.plot(t,p[:,2],color=C[0],lw=2,label="Actual")
a.set_title("Altitude above start (m)"); a.legend(frameon=False)
a=ax[2]; r=np.rad2deg(d["rpy"]); a.plot(t,r[:,0],color=C[0],lw=2,label="Roll"); a.plot(t,r[:,1],color=C[1],lw=2,label="Pitch")
a.set_title("Roll and pitch (degrees)"); a.legend(frameon=False,ncol=2)
a=ax[3]
for i in range(d["pwm"].shape[1]): a.plot(t,d["pwm"][:,i],color=C[i],lw=1.2,label=f"Motor {i+1}")
a.set_ylim(1000,2000); a.set_title("Network output: motor PWM (µs)"); a.set_xlabel("Time (s)"); a.legend(frameon=False,ncol=4 if d["pwm"].shape[1]<=4 else 3,loc="lower center")
a=ax[4]; s=np.array([l["samples"] for l in log])/1e6; a.plot(s,[l["crash_per_1k"] for l in log],color=C[0],lw=2)
a.set_title("Training: crashes per 1,000 control steps"); a.set_xlabel("Training samples (millions)")
fig.tight_layout(h_pad=2); fig.savefig(sys.argv[1] if len(sys.argv)>1 else "nn_drone_results.png",dpi=130)
