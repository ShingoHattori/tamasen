# Experiment (2026-10-13): RGB + aligned stereo depth capture. usage: capture_rgbd.py out.npz N [extended=1]
# RGB 1280x960 + stereo depth aligned to RGB (setOutputSize, TUTORIAL 3.1). Saves npz of N frames + intrinsics.
import sys, json, numpy as np, depthai as dai
sys.path.insert(0, '/home/cely/tamasen/oak')
from oak_common import pick_device, mono_size
out, n = sys.argv[1], int(sys.argv[2]); EXT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
d = pick_device("18443"); mono = mono_size(d)
with dai.Pipeline(d) as p:
    cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    l = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B); r = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    st = p.create(dai.node.StereoDepth).build(left=l.requestOutput(mono, fps=30), right=r.requestOutput(mono, fps=30),
                                             presetMode=dai.node.StereoDepth.PresetMode.DEFAULT)
    st.setDepthAlign(dai.CameraBoardSocket.CAM_A); st.setOutputSize(1280, 960); st.setLeftRightCheck(True); st.setSubpixel(EXT == 0); st.setExtendedDisparity(EXT == 1)
    cq = cam.requestOutput((1280, 960), dai.ImgFrame.Type.BGR888i, fps=30).createOutputQueue(maxSize=8, blocking=False)
    dq = st.depth.createOutputQueue(maxSize=8, blocking=False)
    p.start()
    for _ in range(40): cq.get(); dq.get()
    cs, ds = [], []
    while len(cs) < n + 10:
        cs.append(cq.get()); 
        while dq.has(): ds.append(dq.get())
    while dq.has(): ds.append(dq.get())
    dts = np.array([x.getTimestamp().total_seconds() for x in ds])
    C, D, gaps = [], [], []
    for c in cs[:n]:
        t = c.getTimestamp().total_seconds(); k = int(np.argmin(np.abs(dts - t)))
        gaps.append(abs(dts[k] - t)); C.append(c.getCvFrame()); D.append(ds[k].getFrame())
    print("max rgb-depth time gap ms", round(max(gaps) * 1000, 1))
    tf = c.getTransformation()
    np.savez_compressed(out, color=np.array(C), depth=np.array(D), K=np.array(tf.getIntrinsicMatrix()), dist=np.array(tf.getDistortionCoefficients()))
    print("saved", len(C), "depth valid %", float((np.array(D) > 0).mean() * 100))
