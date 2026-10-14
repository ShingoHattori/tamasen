// Page logic for vision.py. All values from the server are metres / radians; shown as mm / deg.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const mm = (v, d = 1) => v == null ? "—" : (v * 1000).toFixed(d);
const deg = (v, d = 1) => v == null ? "—" : (v * 180 / Math.PI).toFixed(d);
const f = (v, d = 2) => v == null ? "—" : Number(v).toFixed(d);
const esc = (s) => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const STATUS_CLASS = { ok: "ok", pair: "info", ambiguous: "warn", flip: "bad", rejected: "bad" };
const STATUS_JA = { ok: "ok", pair: "もう一方のマーカーとの剛体関係で解を選択", ambiguous: "曖昧", flip: "反転疑い", rejected: "不採用" };
const VALUE_JA = { unset: '<span class="tag no">未入力</span>', sample: '<span class="tag prov">サンプル値・仮定</span>',
  measured: '<span class="tag cal">実測済み</span>', calibrated: '<span class="tag cal">校正済み</span>',
  design: '<span class="tag prov">CAD 設計値・未製作</span>', built: '<span class="tag prov">取り付け済み・未実測</span>' };

async function post(url, body = {}) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  $("msg").textContent = j.message || j.error || "";
  return j;
}

// ------------------------------------------------------------------ 3D view
const el = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
el.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);
const cam = new THREE.PerspectiveCamera(50, 4 / 3, 0.005, 50);
const controls = new OrbitControls(cam, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(0.3, -0.5, 1); scene.add(sun);
const dyn = new THREE.Group(); scene.add(dyn);
let frameMode = "world", fitted = false;

function resize() {
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
  cam.aspect = w / h; cam.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(el); resize();

function setFrame(mode) {
  frameMode = mode; fitted = false;
  if (mode === "world") { cam.up.set(0, 0, 1); cam.position.set(0.35, -0.45, 0.45); controls.target.set(0, 0, 0); }
  else { cam.up.set(0, -1, 0); cam.position.set(0.4, -0.4, -0.3); controls.target.set(0, 0, 0.5); }
  controls.update();
}
document.querySelectorAll('input[name=frame]').forEach(r => r.addEventListener("change", () => setFrame(r.value)));
setFrame("world");

const m4 = (rows) => new THREE.Matrix4().set(...rows.flat());
const inv = (rows) => m4(rows).invert();

function place(obj, M) { obj.matrixAutoUpdate = false; obj.matrix.copy(M); return obj; }

function square(size, color, opacity) {
  const g = new THREE.Group(), h = size / 2;
  const pts = [[-h, h], [h, h], [h, -h], [-h, -h], [-h, h]].map(([x, y]) => new THREE.Vector3(x, y, 0));
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: true, opacity })));
  g.add(new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.3 * opacity, side: THREE.DoubleSide })));
  g.add(new THREE.AxesHelper(size * 0.8));
  const dot = new THREE.Mesh(new THREE.SphereGeometry(size * 0.07), new THREE.MeshBasicMaterial({ color: 0xffff00 }));
  dot.position.set(-h, h, 0); g.add(dot);
  return g;
}

function blockMesh(size, color, opacity, mounts, sizes) {
  const g = new THREE.Group(), geo = new THREE.BoxGeometry(...size);
  g.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.18 * opacity, depthWrite: false })));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color, transparent: true, opacity })));
  g.add(new THREE.AxesHelper(0.045));
  for (const [id, T] of Object.entries(mounts)) g.add(place(square(sizes[id], 0xffffff, 0.8), m4(T)));
  return g;
}

function frustum(K, size, z = 0.15) {
  const [w, h] = size, fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
  const c = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) => new THREE.Vector3((u - cx) / fx * z, (v - cy) / fy * z, z));
  const o = new THREE.Vector3();
  const pts = [o, c[0], o, c[1], o, c[2], o, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]];
  const g = new THREE.Group();
  g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffffff })));
  g.add(new THREE.AxesHelper(0.05));
  return g;
}

function update3d(s) {
  dyn.clear();
  const w = s.world || {}, showAlt = $("showAlt").checked;
  let toView;                       // matrix taking camera-frame poses into the displayed frame
  if (frameMode === "world") {
    if (!w.valid) { $("frameNote").textContent = "world 座標が無効: " + (w.reason || "") + "（world 表示ではマーカーを描かない。カメラ座標表示に切り替えると見える）"; renderer.render(scene, cam); return; }
    toView = inv(w.T_cam_world);
    $("frameNote").textContent = w.mode === "fixed" ? (w.verified ? "" : "固定基準（現フレームで ID 2 による検証なし）") : "毎フレーム追従中（ID 2 が見えない瞬間は world 表示が消える）";
    const grid = new THREE.GridHelper(0.6, 12, 0x335533, 0x223322); grid.rotation.x = Math.PI / 2; dyn.add(grid);
    dyn.add(new THREE.AxesHelper(0.1));
  } else {
    toView = new THREE.Matrix4(); $("frameNote").textContent = "";
    if (w.valid) dyn.add(place(new THREE.AxesHelper(0.1), m4(w.T_cam_world)));
  }
  if (s.intrinsics && s.intrinsics.K) dyn.add(place(frustum(s.intrinsics.K, s.intrinsics.image), toView.clone()));
  const roleColor = { block: 0xffaa00, gripper: 0x66ff66, world_reference: 0x6699ff };
  for (const m of s.markers) {
    if (!m.T_cam_marker || m.status === "rejected") continue;
    const op = m.status === "ok" ? 1 : 0.4;
    dyn.add(place(square(m.size_m, roleColor[m.role] ?? 0xcccccc, op), toView.clone().multiply(m4(m.T_cam_marker))));
    if (showAlt && m.T_cam_marker_alt) dyn.add(place(square(m.size_m, 0xff3333, 0.35), toView.clone().multiply(m4(m.T_cam_marker_alt))));
  }
  const cfg = s.config, mounts = {}, sizes = {};
  for (const [id, mk] of Object.entries(cfg.markers)) if (mk.T_obj_marker) { mounts[id] = mk.T_obj_marker; sizes[id] = mk.size.value; }
  for (const [id, b] of Object.entries(s.block)) {
    dyn.add(place(blockMesh(cfg.block_size.value, id === "0" ? 0x00ffff : id === "3" ? 0xff00ff : 0xcccccc,
      b.status === "ok" || b.status === "pair" ? 1 : 0.4, mounts, sizes), toView.clone().multiply(m4(b.T_cam_obj))));
  }
  const bt = s.block_table;
  if (bt && bt.status === "on_table")
    dyn.add(place(blockMesh(cfg.block_size.value, 0x00dd00, 1, mounts, sizes), toView.clone().multiply(m4(bt.T_cam_obj))));
  if (!fitted && frameMode === "cam") {
    const m = s.markers.find(m => m.T_cam_marker);
    if (m) { controls.target.set(m.T_cam_marker[0][3], m.T_cam_marker[1][3], m.T_cam_marker[2][3]); controls.update(); fitted = true; }
  }
}
(function loop() { renderer.render(scene, cam); requestAnimationFrame(loop); })();

// ------------------------------------------------------------------ tables
function pos(T) { return T ? `(${mm(T[0][3], 1)}, ${mm(T[1][3], 1)}, ${mm(T[2][3], 1)})` : "—"; }
function normal(T) { return T ? `(${f(T[0][2])}, ${f(T[1][2])}, ${f(T[2][2])})` : "—"; }

function renderMarkers(s) {
  const rows = s.markers.map(m => `<tr><td>ID ${m.id} <span class="dim">${m.role}</span></td>
    <td class="l ${STATUS_CLASS[m.status]}">${STATUS_JA[m.status]}${m.reasons.length ? `<br><span class="dim">${m.reasons.map(esc).join("<br>")}</span>` : ""}</td>
    <td>${m.size_m ? mm(m.size_m, 0) : "?"} ${m.size_status === "measured" ? "" : `<span class="warn">${m.size_status}</span>`}</td>
    <td>${pos(m.T_cam_marker)}</td><td>${normal(m.T_cam_marker)}</td><td>${pos(m.T_world_marker)}</td>
    <td>${f(m.reproj_px)}</td><td>${f(m.ambiguity)}</td><td>${f(m.side_px, 0)}</td></tr>`);
  for (const [id, age] of Object.entries(s.not_seen_age_s)) {
    const note = id === "0" ? "（ID 3 の面が見える置き方では正常）" : "";
    rows.push(`<tr><td>ID ${id}</td><td class="l dim" colspan="8">未検出${note}${age == null ? "" : `、最終検出 ${f(age, 1)} s 前（現在位置としては使わない）`}</td></tr>`);
  }
  $("markers").innerHTML = `<table><tr><th>ID</th><th class="l">状態</th><th>一辺 mm</th><th>位置 cam (mm)</th><th>法線 z cam</th>
    <th>位置 world (mm)</th><th>再投影 px</th><th title="IPPE 2解の誤差比。1 に近いほど面の傾きの向きが決まらない">曖昧度</th><th>辺 px</th></tr>${rows.join("")}</table>
    <div class="dim">閾値: 再投影 ≤ ${s.thresholds.max_reproj_px} px、曖昧度 > ${s.thresholds.ambiguous_ratio} で曖昧、
    直前の ok から法線が ${s.thresholds.flip_deg}° 超変化で反転疑い。曖昧・反転疑いは表示するが world 登録などには使わない。平滑化はしていない。</div>`;
}

function renderBlock(s) {
  const lines = [];
  for (const [id, b] of Object.entries(s.block))
    lines.push(`ID ${id} から: <span class="${STATUS_CLASS[b.status]}">${STATUS_JA[b.status]}</span> 中心 cam ${pos(b.T_cam_obj)} / world ${pos(b.T_world_obj)} mm`);
  const bt = s.block_table;
  if (bt) lines.push(bt.status === "on_table"
    ? `<b class="ok">机の面に拘束（緑）</b>: 中心 台座 ${pos(bt.T_base_obj)} mm、${esc(bt.face)}、ID ${bt.markers.join(", ")} の四隅に当てはめ（${f(bt.rms_px)} px、元の傾き ${f(bt.tilt_raw_deg, 1)}°）、静止平滑 ${bt.smoothed_frames} フレーム`
    : `<span class="dim">机の面に拘束: なし（${esc(bt.reason || "")}）</span>`);
  const c = s.compare_0_3;
  if (c) lines.push(`<b>ID 0 と ID 3 の差</b>: 位置 ${mm(c.dt_m)} mm（物体座標 ${c.dt_obj_m.map(v => mm(v)).join(", ")}）、回転 ${deg(c.drot_rad, 2)}°
     ${c.both_ok ? '<span class="ok">（両方とも単独で ok の独立な推定）</span>' : '<span class="warn">（' + Object.entries(c.statuses).map(([k, v]) => "ID " + k + ": " + STATUS_JA[v]).join("、") + '。独立な一致の証拠にはならない）</span>'} <span class="dim">平均はしていない</span>`);
  const unreg = ["0", "3"].filter(id => !s.config.markers[id]?.T_obj_marker);
  if (unreg.length) lines.push(`<span class="warn">ID ${unreg.join(", ")} の取り付けが未確定のため、そのマーカーからのブロック姿勢は出さない（マーカー自体の姿勢は上表）</span>`);
  $("block").innerHTML = lines.join("<br>") || '<span class="dim">ブロック用マーカーが見えていない</span>';
}

function renderGrip(s) {
  const g = s.markers.find(m => m.id === 1), gc = s.config.markers["1"]?.grip;
  $("grip").innerHTML = (g ? `観測: <span class="${STATUS_CLASS[g.status]}">${STATUS_JA[g.status]}</span> マーカー中心 cam ${pos(g.T_cam_marker)} / world ${pos(g.T_world_marker)} mm`
    : '<span class="dim">未検出</span>') +
    `<br>ID 1 → 把持中心: ${gc && gc.status !== "unset" ? VALUE_JA[gc.status] : '<span class="tag no">未校正</span>（把持中心は表示しない）'}
     <br><span class="dim">関節角から計算した手先姿勢は「シミュレーション」欄に別に保持する。カメラ⇔アーム台座が未校正のため両者を同じ座標に重ねない。</span>`;
}

function renderWorld(s) {
  const w = s.world;
  const saved = w.saved ? `保存済み基準: ${w.saved.registered_at}（位置ばらつき ${mm(w.saved.pos_std_m, 2)} mm、回転 ${deg(w.saved.rot_max_dev_rad, 2)}°）` : "保存済み基準なし";
  $("world").innerHTML = `
   状態: <b>${w.mode === "fixed" ? "固定（登録済みの基準を使用）" : "毎フレーム追従（未固定）"}</b>
   ${w.valid ? '<span class="ok">有効</span>' : '<span class="bad">無効</span>'}
   ${w.mode === "fixed" ? (w.verified ? '<span class="ok">現フレームで検証済み</span>' : '<span class="warn">未検証</span>') : ""}
   ${w.reason ? `<br><span class="warn">${esc(w.reason)}</span>` : ""}
   ${w.drift ? `<br>固定基準と現在の ID 2 の差: ${mm(w.drift.dt_m, 2)} mm / ${deg(w.drift.drot_rad, 2)}°` : ""}
   ${w.registering != null ? `<br><span class="info">登録中 ${w.registering}/${w.register_frames} フレーム</span>` : ""}
   <br>${saved}
   <br>${w.card_physically_fixed ? "カード: 物理的に固定済み（申告）" : '<span class="warn">カード: 物理的に未固定（置いてあるだけ）。登録した基準は動けば無効になる</span>'}
   <br><span class="dim">world = ID 2 マーカー座標（原点 = 中心、x/y = 印刷の右/上、z = カード面の外向き）。Z=0 はカード表面で、床面ではない（カード厚み未測定）。
   この基準は暫定 world であり、アーム台座の校正値としては保存しない。</span>
   ${w.message ? `<br><span class="info">${esc(w.message)}</span>` : ""}`;
}

function renderMissing(s) {
  const items = (s.config_error ? [`<span class="bad">${esc(s.config_error)}</span>`] : []).concat(s.missing.map(esc));
  const b = s.config.block_size;
  const mk = Object.entries(s.config.markers).map(([id, m]) => `ID ${id} (${m.role}): 一辺 ${m.size.value ? mm(m.size.value, 0) + " mm" : "?"} ${VALUE_JA[m.size.status]}` +
    (m.role === "block" ? `、取り付け ${VALUE_JA[m.mount_status]}${m.mount_evidence ? ` <span class="dim">${esc(m.mount_evidence)}</span>` : ""}` : "") +
    (m.card_thickness ? `、カード厚み ${VALUE_JA[m.card_thickness.status]}` : ""));
  $("missing").innerHTML = `<ul>${items.map(x => `<li>${x}</li>`).join("")}</ul>
    <div>ブロック ${b.value.map(v => mm(v, 0)).join(" × ")} mm ${VALUE_JA[b.status]}</div><div>${mk.join("<br>")}</div>
    <div class="dim">oak/markers.toml を編集すると起動中のまま反映される。</div>`;
}

function renderIntr(s) {
  const i = s.intrinsics || {};
  if (!i.K) { $("intr").textContent = "未取得"; return; }
  $("intr").innerHTML = `${esc(i.source_text)}<br><span class="dim">${esc(i.camera || "")} ${esc(i.sensor || "")}
    ${i.has_autofocus === false ? "固定焦点（hasAutofocus=false）" : i.has_autofocus ? '<span class="warn">オートフォーカス</span>' : ""}</span><br>
    ${i.calibrated_at ? `校正解像度 ${i.calibrated_at.join("×")} → センサーモード ${i.sensor_mode.join("×")} → ` : ""}処理画像 ${i.image.join("×")}
    ${i.image_matches ? '<span class="ok">（処理画像と一致）</span>' : '<span class="bad">（不一致）</span>'}<br>
    fx = ${f(i.K[0][0])}, fy = ${f(i.K[1][1])}, cx = ${f(i.K[0][2])}, cy = ${f(i.K[1][2])}；歪み ${esc(i.dist_model)} ${i.dist_coeffs.length} 係数<br>
    ${i.naive_fx ? `<span class="dim">EEPROM 値を単純縮小した fx = ${f(i.naive_fx)}（${f(i.naive_fx_error_pct, 1)} %、使用しない）</span>` : ""}`;
}

function renderHeader(s) {
  $("src").textContent = `${s.source === "replay" ? "再生: " + s.replay.dir : "カメラ"} | ${s.fps} | 処理 ${f(s.proc_ms, 0)} ms | 遅延 ${f(s.latency_s * 1000, 0)} ms`;
  const r = s.recorder;
  $("rec").innerHTML = r.recording ? `<span class="bad">● 録画中</span> ${f(r.elapsed_s, 0)}/${r.max_s} s, ${f(r.size_mb, 0)}/${r.max_mb} MB, ${r.written} 枚（取りこぼし ${r.dropped}）`
    : (r.last ? `前回: ${esc(r.last.dir)} ${r.last.written} 枚、取りこぼし ${r.last.dropped}（${esc(r.last.stop_reason)}）` : "録画していない");
}

async function pollState() {
  try {
    const s = await (await fetch("/state.json", { cache: "no-store" })).json();
    if (s.markers) {
      $("jigState").innerHTML = s.jig_mode && s.jig_mode.on ? ` <b class="warn">治具モード中（ID ${s.jig_mode.id} = 治具 ${mm(s.jig_mode.size_m, 0)} mm、ブロック推定なし）</b> ` : " 治具モード OFF ";
      renderHeader(s); renderMarkers(s); renderBlock(s); renderGrip(s); renderWorld(s); renderMissing(s); renderIntr(s); update3d(s);
    }
  } catch (e) { $("src").textContent = "state.json 取得失敗: " + e; }
  setTimeout(pollState, 150);
}
pollState();

// ------------------------------------------------------------------ arm (read-only service)
async function pollArm() {
  try {
    const a = await (await fetch("/arm/state.json", { cache: "no-store" })).json();
    if (a.available === false) $("arm").innerHTML = `<span class="warn">${esc(a.error)}</span><br><span class="dim">rebot/arm_state.py が起動していない。</span>`;
    else {
      const rows = a.joints.map(j => `<tr><td>${j.name}</td><td>${j.pos == null ? "—" : deg(j.pos, 2)}</td><td>${j.vel == null ? "—" : f(j.vel, 3)}</td>
        <td>${j.torq == null ? "—" : f(j.torq, 3)}</td><td>${j.status ?? "—"}</td><td class="${j.age_s != null && j.age_s < 0.5 ? "ok" : "bad"}">${f(j.age_s, 2)}</td></tr>`);
      $("arm").innerHTML = `<div>${esc(a.mode)} — ${esc(a.port)}、${a.ok ? '<span class="ok">応答あり</span>' : '<span class="bad">応答なし / 古い</span>'}</div>
        <table><tr><th>関節</th><th>角度 °</th><th>速度 rad/s</th><th>トルク Nm</th><th>状態コード</th><th>経過 s</th></tr>${rows.join("")}</table>
        <div class="dim">${(a.notes || []).map(esc).join("<br>")}</div>
        <h2>補正の区別</h2><div>${a.compensation ? Object.values(a.compensation).map(esc).join("<br>") : ""}</div>
        <h2>モーター設定（起動時に読み取り）</h2><div class="dim">${a.motor_params ? Object.entries(a.motor_params).map(([n, p]) =>
          `${n}: CAN タイムアウト ${esc(p.can_timeout)}、制御モード ${esc(p.ctrl_mode)}`).join("<br>") : "—"}
          <br>CAN タイムアウト 0 = 無効（指令が途絶えてもモーターは無効化しない）、0 以外 = その時間で無効化（アームが落ちる）。制御モード 1 MIT / 2 位置速度 / 3 速度</div>
        <h2>異常時の動作</h2><div class="dim">${a.faults_policy ? Object.entries(a.faults_policy).map(([k, v]) => `<b>${esc(k)}</b>: ${esc(v)}`).join("<br>") : ""}</div>`;
    }
    renderCtl(a.available === false ? null : a);
  } catch (e) { $("arm").textContent = String(e); }
  setTimeout(pollArm, 500);
}
pollArm();

// ------------------------------------------------------------------ real-arm control (user only)
let armState = null, checkedPath = null;
function renderCtl(a) {
  armState = a;
  const show = a && a.motion_allowed;
  $("armctl").style.display = show ? "" : "none";
  if (!show) return;
  const c = a.control || {};
  $("ctlstate").innerHTML = `状態: <b class="${c.state === "read_only" ? "ok" : "bad"}">${{ read_only: "無効（トルクなし）", holding: "有効・保持中", moving: "移動中" }[c.state] || c.state}</b>
    ${c.motion ? ` ${esc(c.motion.label)} ${Math.round((c.motion.progress || 0) * 100)} %` : ""}<br>${esc(c.message || "")}
    ${c.fault ? `<br><b class="bad">異常: ${esc(c.fault)}</b>` : ""}
    ${c.gripper && c.gripper.mode ? `<br>グリッパー: ${esc(c.gripper.event || "")}、今の隙間 ${mm(c.gripper.gap_now_m, 1)} mm` : (c.gripper_enabled ? "" : "<br>グリッパー: 無効（最初の操作で有効化）")}
    ${c.fb ? `<br>計画との偏差（フィードバック補正後）: 現在 ${f(c.fb.err_max_now, 3)} rad / 手先 ${mm(c.fb.tcp_dev, 1)} mm、
      最大 ${f(c.fb.max_err, 3)} rad / ${mm(c.fb.max_tcp_dev, 1)} mm、補正の積分項 ${(c.fb.I || []).map(v => f(v, 3)).join(", ")} rad` : ""}
    <br><span class="dim">速度上限 ${a.limits.vmax_rad_s} rad/s。有効中（移動・保持とも）は常に 計画−実測 の PI 補正（Kp ${a.limits.kp.join("/")}, Ki ${a.limits.ki.join("/")}, 不感帯 ${a.limits.deadband_rad} rad）。
      偏差 ${a.limits.corridor_rad} rad / 手先 ${mm(a.limits.corridor_m, 0)} mm を超えたら停止。生存確認 ${a.limits.heartbeat_s} s</span>`;
  const fresh = checkedPath && checkedPath.ok && (Date.now() / 1000 - checkedPath.checked_at) < 120;
  $("execPath").disabled = !(fresh && c.state === "holding" && !c.fault);
}
setInterval(() => {               // dead-man: only while this page is visible and the arm is engaged
  if (armState && armState.motion_allowed && armState.control && armState.control.state !== "read_only" && document.visibilityState === "visible")
    fetch("/arm/api/heartbeat", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
}, 200);
$("engage").onclick = () => { const t = prompt("実機の 6 関節を有効化し、現在位置で保持します。アーム周辺に人・物がないことを確認して MOVE と入力"); if (t) post("/arm/api/engage", { confirm: t }); };
$("stopArm").onclick = () => { if (autoRun) autoRun.abort = true; post("/arm/api/stop"); };
async function simulateAndGo(target, what) {
  const p = await post("/sim/api/check_path", { target });
  checkedPath = p;
  $("pathres").innerHTML = p.ok ? `<span class="ok">${esc(p.label)}: ${esc(p.message)}</span>` : `<span class="bad">${what}へ戻る経路が不可: ${esc(p.message || "")}</span>`;
  if (!p.ok) return;
  if (confirm(`${what}へ戻します（シミュレーション済み、所要 ${f(p.duration_s, 1)} s、最大関節変化 ${deg(p.max_step_rad, 1)}°）。経路上に物がないことを目視で確認しましたか？`))
    post("/arm/api/execute", { path_id: p.id });
}
$("goZero").onclick = () => simulateAndGo("zero", "ゼロ姿勢");
$("goHome").onclick = async () => {
  const p = await post("/sim/api/check_path", { target: "engage_pose" });
  checkedPath = p;
  $("pathres").innerHTML = p.ok ? `<span class="ok">${esc(p.label)}: ${esc(p.message)}</span>` : `<span class="bad">初期位置へ戻る経路が不可: ${esc(p.message || "")}</span>`;
  if (!p.ok) return;
  if (confirm(`初期位置（有効化した姿勢）へ戻します（シミュレーション済み、所要 ${f(p.duration_s, 1)} s、最大関節変化 ${deg(p.max_step_rad, 1)}°）。経路上に物がないことを目視で確認しましたか？`))
    post("/arm/api/execute", { path_id: p.id });
};
$("ackFault").onclick = () => { if (confirm("異常の原因を確認し、取り除きましたか？（原因が続いている場合は解除されません）")) post("/arm/api/ack_fault"); };
$("release").onclick = () => { const t = prompt("トルクを切ると J2/J3 が自重で落ちます。アームを手で支えてから RELEASE と入力"); if (t) post("/arm/api/release", { confirm: t }); };
document.querySelectorAll("[data-path]").forEach(b => b.onclick = async () => {
  checkedPath = await post("/sim/api/check_path", { target: b.dataset.path });
  const p = checkedPath;
  $("pathres").innerHTML = (p.ok ? `<span class="ok">${esc(p.label)}: ${esc(p.message)}</span>` : `<span class="bad">${esc(p.label || "")} ${esc(p.message || "")}</span>`) +
    (p.duration_s ? `<br>所要 ${f(p.duration_s, 1)} s、最大関節変化 ${deg(p.max_step_rad, 1)}°、床との最小距離 ${mm(p.min_floor_clearance_m, 0)} mm${p.min_floor_clearance_m >= 0.05 ? " 以上" : ""}、
      シミュレーション上の追従誤差 ${f(p.sim_track_max_rad, 3)} rad。実行できるのはこの確認から 120 s 以内、始点が現在の姿勢と一致する場合だけ` : "");
});
$("execPath").onclick = () => {
  if (!checkedPath) return;
  if (confirm(`実機で「${checkedPath.label}」を実行します（最大関節変化 ${deg(checkedPath.max_step_rad, 1)}°）。経路上にカメラスタンドや物がないことを目視で確認しましたか？`))
    post("/arm/api/execute", { path_id: checkedPath.id });
};
$("graspReal").onclick = () => post("/arm/api/grasp");

// ------------------------------------------------------------------ automatic grasp sequence (started by the user)
// Every move is simulated first (check_path) and only a clean path is executed; any fault, refused path,
// failed alignment or missed grip ends the sequence with the arm holding where it is.
const AUTO_ALIGN_MM = 3, AUTO_ALIGN_MAX_MM = 15, AUTO_ALIGN_TRIES = 3;
let autoRun = null;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const armNow = async () => (await fetch("/arm/state.json", { cache: "no-store" })).json();
function autoLog(t) { $("autoLog").textContent += `${new Date().toLocaleTimeString()} ${t}\n`; }
function autoCheck() { if (autoRun.abort) throw new Error("中止した"); }
async function autoMove(target, label) {
  autoCheck();
  const p = await post("/sim/api/check_path", { target });
  checkedPath = p;
  if (!p.ok) throw new Error(`${label}: 経路が通らない — ${p.message || ""}`);
  autoLog(`${label}: 経路シミュレーション OK（${f(p.duration_s, 1)} s、床 ${mm(p.min_floor_clearance_m, 0)} mm、顎 ${mm(p.gap_m, 0)} mm）→ 実行`);
  const r = await post("/arm/api/execute", { path_id: p.id });
  if (r.ok === false) throw new Error(`${label}: 実行を拒否された — ${r.message || ""}`);
  const t0 = Date.now();
  for (;;) {
    await sleep(200);
    const c = (await armNow()).control || {};
    if (c.fault) throw new Error(`${label}: 異常 — ${c.fault}`);
    if (c.state !== "moving" && Date.now() - t0 > 600) {
      if (c.state !== "holding") throw new Error(`${label}: 状態 ${c.state}`);
      autoLog(`${label}: ${c.message || "完了"}`);
      return;
    }
    if (autoRun.abort) { await post("/arm/api/stop"); throw new Error("中止した（停止を送った）"); }
  }
}
async function autoAlign(target, label) {
  let applied = null;
  for (let i = 0; i < AUTO_ALIGN_TRIES; i++) {
    autoCheck();
    await sleep(800);                                     // let the arm settle before looking at ID 1
    const r = await post("/sim/api/visual_correct", {});
    if (!r.ok) throw new Error(`${label}の位置合わせ: ${r.message || ""}`);
    const o = r.correction.offset_mm, n = Math.hypot(...o);
    // the offset is the FK-vs-camera error at this pose; after moving by it, the next measurement should
    // repeat it (the error of the calibration), so the residual is its change, not its size
    const resid = applied ? Math.hypot(o[0] - applied[0], o[1] - applied[1], o[2] - applied[2]) : n;
    autoLog(`${label}の位置合わせ ${i + 1}: ずれ ${n.toFixed(1)} mm（${o.map(v => v.toFixed(1)).join(", ")}）、残り ${resid.toFixed(1)} mm`);
    if (n > AUTO_ALIGN_MAX_MM) throw new Error(`${label}: ずれ ${n.toFixed(1)} mm が自動の上限 ${AUTO_ALIGN_MAX_MM} mm を超える（校正・観測を確認）`);
    if (resid < AUTO_ALIGN_MM) return;
    applied = o;
    await autoMove(target, `${label}（補正後）`);
  }
  throw new Error(`${label}: 位置合わせが ${AUTO_ALIGN_TRIES} 回で収まらない`);
}
async function autoGripper(body, label, done, timeout_ms) {
  autoCheck();
  const r = await post("/arm/api/gripper", body);
  if (r.ok === false) throw new Error(`${label}: ${r.message || ""}`);
  const t0 = Date.now();
  for (;;) {
    await sleep(200);
    const c = (await armNow()).control || {};
    if (c.fault) throw new Error(`${label}: 異常 — ${c.fault}`);
    const g = c.gripper || {};
    const v = done(g);
    if (v) { autoLog(`${label}: ${g.event || ""}`); return v; }
    if (Date.now() - t0 > timeout_ms) throw new Error(`${label}: ${timeout_ms / 1000} s で終わらない（${g.event || ""}）`);
    autoCheck();
  }
}
$("autoGrasp").onclick = async () => {
  if (autoRun) return;
  const a = await armNow(), c = a.control || {};
  if (c.state !== "holding" || c.fault) { $("autoLog").textContent = "アームが有効・保持中（異常なし）のときだけ開始できる"; return; }
  const t = prompt("ブロックを掴んで有効化時の姿勢へ戻るまでを自動で行います。各移動はシミュレーションで確認してから実行し、"
    + "異常・位置合わせ失敗・つかみ損ねで止まります。アーム周辺とカメラスタンドを確認して GRASP と入力");
  if (t !== "GRASP") return;
  autoRun = { abort: false };
  $("autoLog").textContent = "";
  $("autoGrasp").disabled = true;
  try {
    await post("/sim/api/placement", { source: "observed" });       // plan from what the camera sees now
    await sleep(1500);
    autoLog("把持候補を計算中…");
    const pl = await post("/sim/api/plan");
    const st = await (await fetch("/sim/api/state", { cache: "no-store" })).json();
    const cand = ((st.plan || {}).candidates || []).find(x => x.ok);
    if (!cand) throw new Error("条件を満たす把持候補がない — " + (pl.message || ""));
    const width = Math.round((cand.width_m - 0.01) * 1000), open = Math.round(cand.open_gap_m * 1000);
    autoLog(`候補: ${cand.name}（物の幅 ${width} mm、接近時の開口 ${open} mm）`);
    await autoMove("pre", "接近前");
    await autoAlign("pre", "接近前");
    await autoGripper({ action: "open", gap_mm: open }, `開く（${open} mm）`, g => g.gap_now_m != null && g.gap_now_m * 1000 >= open - 3, 8000);
    await autoMove("near", "近接");
    await autoAlign("near", "近接");
    await autoMove("grasp", "把持姿勢（低速）");
    const res = await autoGripper({ action: "grip", width_mm: width }, "挟む", g => g.result, 10000);
    if (res !== "holding") {
      await autoGripper({ action: "release", gap_mm: open }, `離す（${open} mm まで開く）`, g => g.gap_now_m != null && g.gap_now_m * 1000 >= open - 3, 8000);
      throw new Error(`挟む: ${res === "missed" ? "つかみ損ね" : "閉じない"}。顎を開いてその場で保持（位置を確かめてやり直す）`);
    }
    await autoMove("lift", "持ち上げ");
    const g2 = ((await armNow()).control || {}).gripper || {};
    if (g2.result !== "holding") throw new Error("持ち上げ後に把持していない（落とした疑い）");
    await autoMove("engage_pose", "有効化時の姿勢へ");
    autoLog("完了: ブロックを把持したまま有効化時の姿勢で保持中");
  } catch (e) {
    autoLog("停止: " + e.message);
  } finally {
    autoRun = null;
    $("autoGrasp").disabled = false;
  }
};
$("visCorrect").onclick = async () => {
  $("pathres").innerHTML = '<span class="info">ID 1 を観測中（約 1 秒、動かさないで）…</span>';
  const r = await post("/sim/api/visual_correct", {});
  checkedPath = null;
  $("pathres").innerHTML = r.ok ? `<span class="ok">${esc(r.message)}</span>` : `<span class="bad">${esc(r.message || "")}</span>`;
};
let gripFromPlan = null;          // plan time the inputs were last filled from
function gripCmd(action, extra, what) {
  if (confirm(`グリッパーを${what}。顎の間に指や物がないことを確認しましたか？`)) post("/arm/api/gripper", Object.assign({ action }, extra));
}
$("gripOpen").onclick = () => gripCmd("open", { gap_mm: Number($("gripGap").value) }, `隙間 ${$("gripGap").value} mm まで開きます`);
$("gripGrip").onclick = () => gripCmd("grip", { width_mm: Number($("gripWidth").value) }, `閉じて幅 ${$("gripWidth").value} mm の物を挟みます（トルク制限）`);
$("gripRelease").onclick = () => gripCmd("release", { gap_mm: Number($("gripGap").value) }, `隙間 ${$("gripGap").value} mm まで開いて離します`);
$("previewPath").onclick = () => post("/sim/api/preview");

// ------------------------------------------------------------------ simulator
let simCam = "overview", simGeom = "visual";
function simStream() { $("simimg").src = `/sim/stream.mjpg?cam=${simCam}&geom=${simGeom}&t=${Date.now()}`; }
document.querySelectorAll("[data-cam]").forEach(b => b.onclick = () => { simCam = b.dataset.cam; simStream(); });
document.querySelectorAll("[data-geom]").forEach(b => b.onclick = () => { simGeom = b.dataset.geom; simStream(); });
document.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => post("/sim/api/mode", { mode: b.dataset.mode }));
document.querySelectorAll("[data-place]").forEach(b => b.onclick = () => post("/sim/api/placement", { source: b.dataset.place }));
$("plan").onclick = () => post("/sim/api/plan");
{ // drag to orbit the MuJoCo view
  const img = $("simimg"); let drag = null, view = { azimuth: 135, elevation: -30, distance: 1.0 };
  img.onmousedown = (e) => { drag = { x: e.clientX, y: e.clientY, az: view.azimuth, el: view.elevation }; e.preventDefault(); };
  window.addEventListener("mouseup", () => drag = null);
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    view.azimuth = drag.az - (e.clientX - drag.x) * 0.4; view.elevation = Math.max(-89, Math.min(10, drag.el - (e.clientY - drag.y) * 0.3));
    post("/sim/api/view", view);
  });
  img.onwheel = (e) => { view.distance = Math.max(0.2, Math.min(3, view.distance * (e.deltaY > 0 ? 1.1 : 0.9))); post("/sim/api/view", view); e.preventDefault(); };
}
$("trial").onclick = () => post("/sim/api/trial");
$("reset").onclick = () => post("/sim/api/reset");
simStream();

function renderSim(s) {
  if (s.available !== false) renderPoses(s.calib_poses);
  if (s.available === false) { $("sim").innerHTML = `<span class="warn">${esc(s.error)}</span><br><span class="dim">sim/sim_server.py が起動していない。</span>`; return; }
  $("simMode").innerHTML = `<b>${s.mode === "sync" ? "現実同期" : "シミュレーション試行"}</b> ${s.mode === "sync" ? "（物理演算なし。観測と関節角を反映するだけ）" : "（ライブ観測による上書きを停止中）"}`;
  $("simPlace").innerHTML = `${esc(s.placement.label)} ${s.placement.calibrated ? '<span class="tag cal">校正済み</span>' : '<span class="tag prov">未校正・模擬</span>'}`;
  const j = (s.joints || []).map(x => `<tr><td>${x.name}</td><td>${deg(x.q, 1)}</td><td>${deg(x.lo, 0)} … ${deg(x.hi, 0)}</td><td class="l">${esc(x.source)}</td></tr>`);
  const sync = s.sync || {};
  const plan = s.plan, tr = s.trial;
  if (plan && plan.t !== gripFromPlan) {          // fill the gripper inputs from the first ok candidate
    const okc = (plan.candidates || []).find(c => c.ok);
    if (okc) { $("gripGap").value = Math.round(okc.width_m * 1000); $("gripWidth").value = Math.round((okc.width_m - 0.01) * 1000); }
    gripFromPlan = plan.t;
  }
  $("sim").innerHTML = `
    <div class="dim">${(s.notes || []).map(esc).join("<br>")}</div>
    <h2>入力の状態</h2>
    <div>関節: ${esc(sync.joints || "—")}<br>ブロック: ${esc(sync.block || "—")}</div>
    <h2>関節・顎（シミュレータ内）</h2>
    <table><tr><th>関節</th><th>角度 °</th><th>可動範囲 °</th><th class="l">値の出所</th></tr>${j.join("")}</table>
    <div>顎の開口（内側の隙間）: ${mm(s.jaw?.gap_m)} mm、モデル上の最大 ${mm(s.jaw?.max_gap_model_m)} mm、設定の最大 ${mm(s.jaw?.max_gap_config?.value)} mm ${VALUE_JA[s.jaw?.max_gap_config?.status] || ""}
      <br><span class="dim">${esc(s.jaw?.note || "")}</span></div>
    <div>関節角からの手先（end_link、台座座標）: ${s.fk ? `(${s.fk.p.map(v => mm(v)).join(", ")}) mm` : "—"}
      <br><span class="dim">${esc(s.fk_vs_vision || "")}</span></div>
    <h2>把持候補</h2>${plan ? planTable(plan) : '<span class="dim">未計算</span>'}
    <h2>試行</h2>${tr ? trialView(tr) : '<span class="dim">未実施</span>'}
    <h2>障害物</h2>${(s.obstacles || []).map(o => `${esc(o.name)}: <span class="${o.in_scene ? "ok" : "warn"}">${esc(o.label)}</span> <span class="dim">${esc(o.note)}</span>`).join("<br>") || "—"}
    <h2>実機での把持実行の前提</h2>${s.real_grasp_blockers.length ? `<ul>${s.real_grasp_blockers.map(b => `<li class="warn">${esc(b)}</li>`).join("")}</ul>` : '<span class="ok">満たしている</span>'}
    <h2>物理パラメータ</h2>${physTable(s.physics)}
    <h2>接触</h2><div>${(s.contacts || []).map(esc).join("<br>") || '<span class="dim">なし</span>'}</div>`;
}
function planTable(p) {
  const rows = p.candidates.map(c => `<tr><td>${c.name}</td><td>${mm(c.width_m)}</td><td class="${c.ok ? "ok" : "bad"}">${c.ok ? "可" : "不可"}</td><td class="l">${c.reasons.map(esc).join("; ")}</td></tr>`);
  return `<div>配置: ${esc(p.basis_label)}</div><table><tr><th>候補</th><th>必要開口 mm</th><th>判定</th><th class="l">理由</th></tr>${rows.join("")}</table>`;
}
function trialView(t) {
  return `<div>${t.running ? '<span class="info">実行中</span>' : ""} 段階: ${esc(t.phase)} — 結果: <b class="${t.success ? "ok" : t.done ? "bad" : ""}">${esc(t.result || "")}</b></div>
    <div>配置の根拠: ${esc(t.basis_label)}</div><div class="dim">${(t.log || []).slice(-8).map(esc).join("<br>")}</div>
    <div class="warn">シミュレーションでの把持成功は、実機で同じ把持力で成功することを意味しない。</div>`;
}
function physTable(p) {
  if (!p) return "—";
  return `<table>${Object.entries(p).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(JSON.stringify(v.value))}</td><td class="l">${VALUE_JA[v.status] || esc(v.status)}</td></tr>`).join("")}</table>`;
}
async function pollSim() {
  try { renderSim(await (await fetch("/sim/api/state", { cache: "no-store" })).json()); } catch (e) { $("sim").textContent = String(e); }
  setTimeout(pollSim, 400);
}
pollSim();

// ------------------------------------------------------------------ buttons / files
$("snap").onclick = async () => { await post("/api/snapshot"); loadFiles(); };
$("recStart").onclick = () => post("/api/record/start");
$("recStop").onclick = async () => { await post("/api/record/stop"); setTimeout(loadFiles, 1500); };
$("wReg").onclick = () => post("/api/world/register");
$("wRel").onclick = () => post("/api/world/release");
$("wUse").onclick = () => post("/api/world/use_fixed");
async function loadFiles() {
  const fs = await (await fetch("/api/files")).json();
  $("files").innerHTML = fs.map(x => `<div>${x.kind === "snapshots" ? "静止画" : "録画"} ${x.name}: ${x.files.map(n =>
    `<a href="/files/${x.kind}/${x.name}/${n}" target="_blank">${n}</a>`).join(" ")}</div>`).join("") || '<span class="dim">なし</span>';
}
loadFiles();

// ------------------------------------------------------------------ eye-to-hand calibration
const today = new Date().toISOString().slice(0, 10).replace(/-/g, "");
$("heSession").value = today;
$("genPoses").onclick = () => post("/sim/api/calib_poses");
$("genPosesWS").onclick = async () => { const r = await post("/sim/api/calib_poses", { mode: "workspace" }); $("heMsg").innerHTML = r.ok ? `<span class="ok">${esc(r.message)}</span>` : `<span class="bad">${esc(r.message || "")}</span>`; };
$("previewPoses").onclick = () => post("/sim/api/preview_calib");
$("heSolve").onclick = async () => { await post("/api/handeye/solve", { session: $("heSession").value }); loadHE(); };
$("heAdopt").onclick = async () => { if (confirm("この校正結果を採用し、シミュレーターの配置を校正済みにします。よろしいですか？")) { await post("/api/handeye/adopt", { session: $("heSession").value }); loadHE(); } };
$("gripCap").onclick = () => post("/api/handeye/grip", { frames: 30 });
$("jigOn").onclick = () => post("/api/handeye/jig_mode", { on: true });
$("jigOff").onclick = () => post("/api/handeye/jig_mode", { on: false });
let lastPoses = null;
let recorded = {};      // pose index -> [sample ids]
function renderPoses(cp) {
  if (!cp) { $("calibPoses").innerHTML = '<span class="dim">未生成</span>'; return; }
  const key = cp.t + JSON.stringify(recorded);
  if (lastPoses === key) return; lastPoses = key;
  $("calibPoses").innerHTML = `<table><tr><th>#</th><th class="l">用途</th><th class="l">姿勢（end_link の回転 / 並進）</th><th class="l">静的チェック</th><th></th></tr>` +
    cp.poses.map(p => `<tr><td>${p.i}</td><td class="l">${p.role === "eval" ? '<span class="info">評価用</span>' : "校正用"}</td><td class="l">${esc(p.label)}</td>
      <td class="l ${p.ok ? "ok" : "bad"}">${p.ok ? "ok" : esc(p.reasons.join("; "))}${recorded[p.i] ? ` <b class="ok">記録済み #${recorded[p.i].join(", #")}</b>` : ""}</td>
      <td>${p.ok ? `<button data-calsim="${p.i}">経路をシミュレート</button><button data-calcap="${p.i}" data-role="${p.role}" data-label="${esc(p.label)}">サンプル記録</button>` : ""}</td></tr>`).join("") + "</table>" +
    `<div class="dim">${cp.folded_centre_lifted ? "現在の姿勢が畳まれていたため、同じ向きで 10 cm 持ち上げた姿勢を中心にした。" : ""}実行は上の実機パネル（依頼者の操作）。ID 1 が見えるかは記録時に判定する。</div>`;
  document.querySelectorAll("[data-calsim]").forEach(b => b.onclick = async () => {
    checkedPath = await post("/sim/api/check_path", { target: "calib:" + b.dataset.calsim });
    $("pathres").innerHTML = (checkedPath.ok ? `<span class="ok">${esc(checkedPath.label)}: ${esc(checkedPath.message)}</span>` : `<span class="bad">${esc(checkedPath.message || "")}</span>`);
  });
  document.querySelectorAll("[data-calcap]").forEach(b => b.onclick = async () => {
    $("heMsg").innerHTML = `<span class="info">姿勢 ${b.dataset.calcap} を記録中（約 3 秒、動かさないで）…</span>`;
    const r = await post("/api/handeye/capture", { session: $("heSession").value, role: b.dataset.role, label: "pose " + b.dataset.calcap + " " + b.dataset.label });
    $("heMsg").innerHTML = r.ok ? `<span class="ok">✔ ${esc(r.message)}</span>` :
      (r.id != null ? `<span class="warn">△ ${esc(r.message)}</span>` : `<span class="bad">✘ 記録失敗: ${esc(r.message || "")}</span>`);
    lastPoses = null; loadHE();
  });
}
const heChecked = new Set();
async function heDelete(ids, all = false) {
  if (!confirm(all ? "このセッションのサンプルをすべて削除します（退避ファイルに残ります）。よろしいですか？" : `サンプル #${ids.join(", #")} を削除します。よろしいですか？`)) return;
  const r = await post("/api/handeye/delete", { session: $("heSession").value, ids, all });
  $("heMsg").innerHTML = r.ok ? `<span class="ok">${esc(r.message)}</span>` : `<span class="bad">${esc(r.message || "")}</span>`;
  heChecked.clear(); lastPoses = null; loadHE();
}
$("heDelSel").onclick = () => heChecked.size ? heDelete([...heChecked]) : ($("heMsg").textContent = "削除するサンプルにチェックを入れてください");
$("heDelAll").onclick = () => heDelete([], true);
async function loadHE() {
  try {
    const r = await (await fetch("/api/handeye/sessions")).json();
    const cur = r.sessions.find(x => x.session === $("heSession").value);
    const res = cur && cur.result;
    const list = (cur && cur.sample_list) || [];
    const rec = {};
    for (const x of list) { const m = /^pose (\d+)/.exec(x.label || ""); if (m) (rec[m[1]] = rec[m[1]] || []).push(x.id); }
    if (JSON.stringify(rec) !== JSON.stringify(recorded)) { recorded = rec; }
    const nc = list.filter(x => x.role === "calib").length, ne = list.filter(x => x.role === "eval").length;
    $("heSamples").innerHTML = `<b>記録済みサンプル: 校正用 ${nc} 個（6 個以上必要）、評価用 ${ne} 個（1 個以上必要）</b>` + (list.length ?
      `<table><tr><th></th><th>#</th><th class="l">用途</th><th class="l">姿勢</th><th>ID 1 検出</th><th>角点の揺れ px</th><th class="l">時刻</th><th></th></tr>` +
      list.map(x => `<tr><td><input type="checkbox" class="heSel" value="${x.id}" ${heChecked.has(x.id) ? "checked" : ""}></td><td>${x.id}</td><td class="l">${x.role === "eval" ? "評価用" : "校正用"}</td><td class="l">${esc(x.label)}</td>
        <td>${x.n_frames} フレーム</td><td class="${x.corner_std_px > 0.5 ? "warn" : ""}">${f(x.corner_std_px, 3)}${x.corner_std_px > 0.5 ? "（揺れ大）" : ""}</td><td class="l">${esc(x.time)}</td>
        <td><button class="heDelOne" data-id="${x.id}">削除</button></td></tr>`).join("") + "</table>" : "");
    document.querySelectorAll(".heSel").forEach(cb => cb.onchange = () => cb.checked ? heChecked.add(+cb.value) : heChecked.delete(+cb.value));
    document.querySelectorAll(".heDelOne").forEach(b => b.onclick = () => heDelete([+b.dataset.id]));
    $("heResult").innerHTML = `サンプル ${cur ? cur.samples : 0} 個` + (res ? `<br>結果: <b class="${res.ok ? "ok" : "bad"}">${esc(res.message || "")}</b>
      （校正 ${res.n_calib}、評価 ${res.n_eval}）<br>回転の多様性: ${esc(res.diversity ? res.diversity.message : "")}
      ${res.holdout && res.holdout.n ? `<br>評価用姿勢: 再投影 平均 ${f(res.holdout.reproj_px_mean)} / 最大 ${f(res.holdout.reproj_px_max)} px、位置 中央値 ${f(res.holdout.pos_err_mm_median, 1)} mm（画像面内 ${f(res.holdout.lateral_err_mm_median, 1)} mm）、
        回転 中央値 ${f(res.holdout.rot_err_deg_median)}°（参考: 単一マーカー姿勢のノイズを含む）` : ""}
      ${res.sd ? `<br>推定の標準偏差: camera↔base 並進 ${res.sd.cam_base_t_mm.map(v => f(v, 1)).join("/")} mm、回転 ${res.sd.cam_base_rot_deg.map(v => f(v, 2)).join("/")}°；end_link↔ID 1 並進 ${res.sd.ee_m1_t_mm.map(v => f(v, 1)).join("/")} mm` : ""}` : "") +
      `<br>採用中: ${r.active ? `${esc(r.active.session)}（${esc(r.active.adopted_at)}）` : '<span class="warn">なし（camera↔base 未校正）</span>'}`;
  } catch (e) { $("heResult").textContent = String(e); }
}
loadHE(); setInterval(loadHE, 5000);
