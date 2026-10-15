// 3D ビュー（marker 座標で表示、mm）。組み立てた形状・差分部品（赤半透明）・選択部品・マーカー・カメラ・作図平面・深度点群。
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Vec3 } from "../types";
import { mat4InvRigid, mat4Apply } from "./geo";
import { api } from "./api";
import { m4 } from "./camera";
import { currentModel, modelVersion } from "./model";
import { addCorrespondence } from "./imageView";
import { store } from "./store";

export class View3D {
  readonly el: HTMLElement;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(40, 1, 1, 50000);
  private controls: OrbitControls | null = null;
  private objGroup = new THREE.Group(); // matrix = T_marker_obj
  private modelMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ color: 0xb8c4d0, metalness: 0.1, roughness: 0.7, side: THREE.DoubleSide }));
  private partsGroup = new THREE.Group();
  private markerGroup = new THREE.Group();
  private camsGroup = new THREE.Group();
  private planeGroup = new THREE.Group();
  private pointsGroup = new THREE.Group();
  private builtVersion = -1;
  private builtSel: string | null = "";
  private markerKey = "";
  private depthKey = "";
  private raf = 0;
  private downAt: [number, number] | null = null;
  private framed = false;

  constructor(el: HTMLElement) {
    this.el = el;
    el.classList.add("view3d");
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      this.renderer.setClearColor(0x1b1e24);
      el.appendChild(this.renderer.domElement);
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.addEventListener("change", () => this.render());
    } catch {
      el.textContent = "WebGL が使えないため 3D ビューを表示できません";
    }
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(250, -300, 250);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0xffffff, 2.0);
    dl.position.set(300, -200, 500);
    this.scene.add(dl);
    const dl2 = new THREE.DirectionalLight(0xffffff, 0.7);
    dl2.position.set(-300, 300, -200);
    this.scene.add(dl2);
    this.objGroup.matrixAutoUpdate = false;
    this.objGroup.add(this.modelMesh, this.partsGroup, this.planeGroup, this.pointsGroup);
    this.scene.add(this.objGroup, this.markerGroup, this.camsGroup);
    const grid = new THREE.GridHelper(400, 20, 0x3a3f48, 0x2a2e35);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);
    new ResizeObserver(() => this.render()).observe(el);
    if (this.renderer) {
      const c = this.renderer.domElement;
      c.addEventListener("pointerdown", (e) => (this.downAt = [e.clientX, e.clientY]));
      c.addEventListener("pointerup", (e) => {
        if (this.downAt && Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]) < 4) this.onClick(e);
        this.downAt = null;
      });
    }
    store.on(() => this.render());
  }

  render(): void {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  resetView(): void {
    this.framed = false;
    this.render();
  }

  private draw(): void {
    if (!this.renderer) return;
    const W = Math.max(1, this.el.clientWidth),
      H = Math.max(1, this.el.clientHeight);
    const s = this.renderer.getSize(new THREE.Vector2());
    if (s.x !== W || s.y !== H) {
      this.renderer.setSize(W, H);
      this.camera.aspect = W / H;
      this.camera.updateProjectionMatrix();
    }
    const p = store.project;
    if (p) {
      this.objGroup.matrix.copy(m4(p.T_marker_obj));
      this.objGroup.matrixWorldNeedsUpdate = true;
      this.updateModel();
      this.updateMarker();
      this.updateCameras();
      this.updatePlane();
      this.updateDepth();
    }
    this.objGroup.visible = !!p;
    this.renderer.render(this.scene, this.camera);
  }

  private updateModel(): void {
    const m = currentModel();
    const ver = modelVersion();
    const sel = store.ui.selPart;
    if (ver === this.builtVersion && sel === this.builtSel) return;
    this.builtVersion = ver;
    this.builtSel = sel;
    this.modelMesh.geometry = m?.geometry ?? new THREE.BufferGeometry();
    for (const c of [...this.partsGroup.children]) {
      this.partsGroup.remove(c);
      const mm = c as THREE.Mesh;
      (mm.material as THREE.Material).dispose();
    }
    const parts = store.project?.shape.parts ?? [];
    for (const pr of m?.parts ?? []) {
      if (!pr.geometry) continue;
      const part = parts.find((x) => x.id === pr.id);
      if (part && !part.visible) continue;
      if (pr.op === "subtract") {
        const mesh = new THREE.Mesh(pr.geometry, new THREE.MeshBasicMaterial({ color: 0xff3030, transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide }));
        this.partsGroup.add(mesh);
      }
      if (pr.id === sel && store.ui.showGhost) {
        const ghost = new THREE.Mesh(pr.geometry, new THREE.MeshBasicMaterial({ color: 0xffd000, wireframe: true, transparent: true, opacity: 0.6 }));
        this.partsGroup.add(ghost);
      }
    }
    if (!this.framed && m?.stats) {
      this.framed = true;
      const s = m.stats;
      const size = Math.max(40, ...[0, 1, 2].map((i) => s.bbox_max[i] - s.bbox_min[i]));
      const c = new THREE.Vector3((s.bbox_min[0] + s.bbox_max[0]) / 2, (s.bbox_min[1] + s.bbox_max[1]) / 2, (s.bbox_min[2] + s.bbox_max[2]) / 2).applyMatrix4(this.objGroup.matrix);
      this.controls?.target.copy(c);
      this.camera.position.copy(c.clone().add(new THREE.Vector3(1.6, -2.0, 1.5).multiplyScalar(size)));
      this.controls?.update();
    }
  }

  private updateMarker(): void {
    const size = store.project!.marker.size_mm || 30;
    const key = String(size);
    if (key === this.markerKey) return;
    this.markerKey = key;
    this.markerGroup.clear();
    const sq = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color: 0x111111, side: THREE.DoubleSide }));
    const inner = new THREE.Mesh(new THREE.PlaneGeometry(size * 0.5, size * 0.5), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
    inner.position.set(-size * 0.125, size * 0.125, 0.05);
    const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(size * 1.25, size * 1.25)), new THREE.LineBasicMaterial({ color: 0xffffff }));
    const axes = new THREE.AxesHelper(size * 1.5);
    this.markerGroup.add(sq, inner, border, axes);
  }

  private updateCameras(): void {
    this.camsGroup.clear();
    const p = store.project!;
    for (const img of p.images) {
      const T = img.pose.T_cam_marker;
      const cam = p.cameras[img.camera];
      if (!T || !cam || !cam.K_und?.[0]) continue;
      const K = cam.K_und;
      const camPos = mat4Apply(mat4InvRigid(T), [0, 0, 0]);
      const d = Math.max(20, Math.hypot(camPos[0], camPos[1], camPos[2]) * 0.18);
      const corners = [
        [0, 0],
        [img.width, 0],
        [img.width, img.height],
        [0, img.height],
      ].map(([u, v]) => new THREE.Vector3(((u - K[2]) / K[0]) * d, ((v - K[5]) / K[4]) * d, d));
      const pts: THREE.Vector3[] = [];
      const o = new THREE.Vector3();
      for (let i = 0; i < 4; i++) pts.push(o, corners[i], corners[i], corners[(i + 1) % 4]);
      // 上方向（v 小）を示す三角
      const top = new THREE.Vector3(((img.width / 2 - K[2]) / K[0]) * d, ((-img.height * 0.15 - K[5]) / K[4]) * d, d);
      pts.push(corners[0], top, top, corners[1]);
      const sel = img.id === store.ui.selImage;
      const ls = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: sel ? 0xffd000 : img.pose.locked ? 0x60d060 : 0x7090c0 }));
      ls.matrixAutoUpdate = false;
      ls.matrix.copy(m4(mat4InvRigid(T)));
      this.camsGroup.add(ls);
    }
  }

  private updatePlane(): void {
    this.planeGroup.clear();
    const pl = store.ui.planePreview;
    if (!pl) return;
    const n = new THREE.Vector3(...pl.normal).normalize();
    const x = new THREE.Vector3(...pl.xAxis).normalize();
    const y = n.clone().cross(x);
    const size = (() => {
      const s = currentModel()?.stats;
      return s ? Math.max(60, ...[0, 1, 2].map((i) => s.bbox_max[i] - s.bbox_min[i])) * 1.4 : 150;
    })();
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color: 0x30ff90, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
    const basis = new THREE.Matrix4().makeBasis(x, y, n).setPosition(new THREE.Vector3(...pl.origin));
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(basis);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(size, size)), new THREE.LineBasicMaterial({ color: 0x30ff90 }));
    edge.matrixAutoUpdate = false;
    edge.matrix.copy(basis);
    const ax = new THREE.AxesHelper(size * 0.25);
    ax.matrixAutoUpdate = false;
    ax.matrix.copy(basis);
    this.planeGroup.add(mesh, edge, ax);
  }

  private updateDepth(): void {
    const img = store.image();
    const want = store.ui.depthPoints && img && img.depth && img.pose.T_cam_marker && store.pid;
    const key = want ? `${img!.id}|${img!.pose.T_cam_marker!.join(",")}|${store.project!.T_marker_obj.join(",")}` : "";
    if (key === this.depthKey) return;
    this.depthKey = key;
    this.pointsGroup.clear();
    if (!want) return;
    const T_obj_cam = (() => {
      const T_marker_cam = mat4InvRigid(img!.pose.T_cam_marker!);
      const T_obj_marker = mat4InvRigid(store.project!.T_marker_obj);
      return { T_marker_cam, T_obj_marker };
    })();
    api.depthPoints(store.pid!, img!.id, 4).then(
      (res) => {
        if (this.depthKey !== key) return;
        const r = res as { points?: number[][] | number[]; colors?: number[][] };
        let pts: number[][] = [];
        if (Array.isArray(r.points) && r.points.length && Array.isArray(r.points[0])) pts = r.points as number[][];
        else if (Array.isArray(r.points)) for (let i = 0; i + 2 < r.points.length; i += 3) pts.push([(r.points as number[])[i], (r.points as number[])[i + 1], (r.points as number[])[i + 2]]);
        const pos = new Float32Array(pts.length * 3);
        pts.forEach((pc, i) => {
          const pm = mat4Apply(T_obj_cam.T_marker_cam, pc as Vec3);
          const po = mat4Apply(T_obj_cam.T_obj_marker, pm);
          pos.set(po, i * 3);
        });
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        this.pointsGroup.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xff9040, size: 1.5, sizeAttenuation: false })));
        store.msg(`深度点群 ${pts.length} 点`, "info");
        this.render();
      },
      (e: Error) => store.msg(`深度点群を取得できません: ${e.message}`, "error"),
    );
  }

  /** 対応点のモデル側を 3D ビューで選ぶ */
  private onClick(e: PointerEvent): void {
    const img = store.image();
    if (store.ui.mode !== "align" || store.ui.alignTool !== "corr" || !img) return;
    if (!store.ui.corrPending) {
      store.msg("先に画像上の点をクリックしてください", "info");
      return;
    }
    const r = this.renderer!.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.camera);
    this.scene.updateMatrixWorld(true);
    const hit = rc.intersectObject(this.modelMesh, false)[0];
    if (!hit) {
      store.msg("モデルに当たりません", "error");
      return;
    }
    const local = this.objGroup.worldToLocal(hit.point.clone());
    addCorrespondence(img, [local.x, local.y, local.z]);
  }
}


