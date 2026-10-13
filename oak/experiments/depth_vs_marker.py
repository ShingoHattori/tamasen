# Experiment (2026-10-13): depth-plane normal / distance vs marker IPPE solutions. usage: PYTHONPATH=oak python depth_vs_marker.py rgbd.npz
import sys, numpy as np, cv2, math, pose
from pathlib import Path
import measure_accuracy as M
z=np.load(sys.argv[1]); C,D,K,dist=z['color'],z['depth'],z['K'],z['dist']
cfg=pose.load_config(Path('markers.toml')); det=M.make_detector(cfg)
H,W=D.shape[1:]; uu,vv=np.meshgrid(np.arange(W),np.arange(H)); res={}
for f in range(len(C)):
    c,ids,_=det.detectMarkers(cv2.cvtColor(C[f],cv2.COLOR_BGR2GRAY))
    if ids is None: continue
    for cc,i in zip(c,ids.ravel()):
        i=int(i)
        if i not in (1,2,3): continue
        cc=cc.reshape(4,2); size=cfg.markers[i].size_m; sols=pose.solve_marker(cc,size,K,dist)
        r=res.setdefault(i,{'n':[],'ab':[],'aa':[],'alt':[],'zd':[],'zp':[],'res':[],'amb':[],'np':[],'det':0,'Tn':[]})
        r['det']+=1; r['amb'].append(sols[0][0]/sols[1][0] if len(sols)>1 and sols[1][0]>0 else 0); r['Tn'].append(sols[0][1][:3,2])
        mask=np.zeros((H,W),np.uint8); cen=cc.mean(0); cv2.fillConvexPoly(mask,((cc-cen)*0.85+cen).astype(np.int32),1)
        sel=(mask>0)&(D[f]>0); r['np'].append(int(sel.sum()))
        if sel.sum()<50: continue
        pts=np.column_stack([uu[sel],vv[sel]]).astype(np.float64)
        und=cv2.undistortPoints(pts.reshape(-1,1,2),K,dist).reshape(-1,2); zz=D[f][sel].astype(np.float64)/1000
        P=np.column_stack([und*zz[:,None],zz]); m=P.mean(0); _,_,vt=np.linalg.svd(P-m); n=vt[2]
        if n@m>0: n=-n
        a=[math.degrees(math.acos(np.clip(n@T[:3,2],-1,1))) for _,T in sols]
        r['n'].append(n); r['ab'].append(a[0]); r['aa'].append(a[1] if len(a)>1 else np.nan); r['alt'].append(len(a)>1 and a[1]<a[0])
        r['zd'].append(m[2]); r['zp'].append(sols[0][1][2,3]); r['res'].append(np.abs((P-m)@n).std()*1000)
for i,r in sorted(res.items()):
    Tn=np.array(r['Tn']); tm=Tn.mean(0); tm/=np.linalg.norm(tm); tdev=np.degrees(np.arccos(np.clip(Tn@tm,-1,1)))
    print(f"ID {i}: detected {r['det']}, depth pts/frame median {int(np.median(r['np']))}, IPPE ratio median {np.median(r['amb']):.2f}, marker-normal spread median {np.median(tdev):.2f}° max {tdev.max():.1f}°")
    if not r['n']: print('   no depth inside marker'); continue
    N=np.array(r['n']); nm=N.mean(0); nm/=np.linalg.norm(nm); dev=np.degrees(np.arccos(np.clip(N@nm,-1,1)))
    zd,zp=np.array(r['zd']),np.array(r['zp'])
    print(f"   depth plane: frames {len(N)}, resid {np.median(r['res']):.1f} mm, normal spread median {np.median(dev):.2f}° max {dev.max():.2f}°")
    print(f"   angle to IPPE best {np.nanmedian(r['ab']):.1f}°, alt {np.nanmedian(r['aa']):.1f}°; depth prefers ALT in {np.mean(r['alt'])*100:.0f}%")
    print(f"   z: depth {zd.mean()*1000:.1f}±{zd.std()*1000:.2f} mm, PnP {zp.mean()*1000:.1f}±{zp.std()*1000:.2f} mm, ratio {np.mean(zd/zp):.4f}")
# normals relative to the ID 2 card (table) normal: IPPE vs depth
if 2 in res and 3 in res:
    n2 = np.array(res[2]['Tn']).mean(0); n2 /= np.linalg.norm(n2)
    n3_ippe = np.array(res[3]['Tn']).mean(0); n3_ippe /= np.linalg.norm(n3_ippe)
    print(f"ID3 vs table(ID2 IPPE normal): IPPE {math.degrees(math.acos(n2@n3_ippe)):.1f} deg", end="")
    if res[3]['n']:
        n3d = np.array(res[3]['n']).mean(0); n3d /= np.linalg.norm(n3d)
        print(f", depth {math.degrees(math.acos(np.clip(n2@n3d,-1,1))):.1f} deg")
    # box top plane from depth: ring around the block top
    f = 0; c, ids, _ = det.detectMarkers(cv2.cvtColor(C[f], cv2.COLOR_BGR2GRAY))
