import ARKit
import AVFoundation
import SceneKit
import SwiftUI

@main
struct TamasenScanApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

struct PreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let device: AVCaptureDevice?

    final class V: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var pl: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var coord: AVCaptureDevice.RotationCoordinator?
        var obs: NSKeyValueObservation?
    }

    func makeUIView(context: Context) -> V {
        let v = V()
        v.pl.session = session
        v.pl.videoGravity = .resizeAspect
        v.backgroundColor = .black
        return v
    }

    func updateUIView(_ v: V, context: Context) {
        guard let device, v.coord?.device != device else { return }
        let c = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: v.pl)
        v.coord = c
        v.obs = c.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]) { c, _ in
            DispatchQueue.main.async {
                let a = c.videoRotationAngleForHorizonLevelPreview
                NSLog("TSCAN preview angle %f conn=%d", a, v.pl.connection == nil ? 0 : 1)
                if let conn = v.pl.connection, conn.isVideoRotationAngleSupported(a) { conn.videoRotationAngle = a }
            }
        }
    }
}

struct ARPreview: UIViewRepresentable {
    let session: ARSession
    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView()
        v.session = session
        v.automaticallyUpdatesLighting = false
        return v
    }
    func updateUIView(_ v: ARSCNView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var cap = Capture()
    @AppStorage("cam") private var cam = CamKind.lidar.rawValue
    @AppStorage("fps") private var fps = 10.0
    @AppStorage("filter") private var filter = false
    @AppStorage("lock") private var lock = true
    @State private var sessions: [String] = []
    @State private var autoDone = false

    var body: some View {
        HStack(spacing: 0) {
            ZStack(alignment: .topLeading) {
                if cap.usingAR {
                    ARPreview(session: cap.ar)
                } else {
                    PreviewView(session: cap.session, device: cap.device)
                }
                if let img = cap.depthImage {
                    Image(uiImage: img).resizable().interpolation(.none).scaledToFit()
                        .frame(width: 256).opacity(0.85).border(.white).padding(8)
                }
            }
            controls.frame(width: 300).padding().background(Color(white: 0.1))
        }
        .ignoresSafeArea()
        .preferredColorScheme(.dark)
        .onAppear {
            let args = ProcessInfo.processInfo.arguments
            if let i = args.firstIndex(of: "-cam"), i + 1 < args.count, CamKind(rawValue: args[i + 1]) != nil { cam = args[i + 1] }
            AVCaptureDevice.requestAccess(for: .video) { _ in
                cap.recordFPS = fps; cap.filtering = filter; cap.lockAEAF = lock
                cap.configure(CamKind(rawValue: cam) ?? .lidar)
            }
            refresh()
            // 試験用: 起動引数 -autorecord <秒> で 2 秒後に記録し、指定秒で止める
            let a = ProcessInfo.processInfo.arguments
            if !autoDone, let i = a.firstIndex(of: "-autorecord"), i + 1 < a.count, let sec = Double(a[i + 1]) {
                autoDone = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { cap.start() }
                DispatchQueue.main.asyncAfter(deadline: .now() + 2 + sec) { cap.stop() }
            }
        }
        .onChange(of: cap.recording) { _, r in if !r { refresh() } }
    }

    var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("TamasenScan").font(.title2.bold())
            Picker("カメラ", selection: $cam) {
                ForEach(CamKind.allCases) { Text($0.label).tag($0.rawValue) }
            }
            .pickerStyle(.segmented)
            .disabled(cap.recording)
            .onChange(of: cam) { _, v in cap.configure(CamKind(rawValue: v) ?? .lidar) }
            Text(cap.formatText).font(.caption).foregroundStyle(.secondary)
            Text(String(format: "中央の深度 %.0f mm  有効 %.0f%%", cap.centerDepthMM, cap.validPct)).font(.callout.monospacedDigit())
            Stepper(String(format: "記録 %.0f fps", fps), value: $fps, in: 2...30, step: 1)
                .disabled(cap.recording).onChange(of: fps) { _, v in cap.recordFPS = v }
            Toggle("深度フィルタ（穴埋め）", isOn: $filter).onChange(of: filter) { _, v in cap.setFiltering(v) }
            Toggle("記録中は露出・焦点を固定", isOn: $lock).disabled(cap.recording).onChange(of: lock) { _, v in cap.lockAEAF = v }
            Button {
                cap.recording ? cap.stop() : cap.start()
            } label: {
                Text(cap.recording ? "停止" : "記録開始").font(.title3.bold()).frame(maxWidth: .infinity).padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent).tint(cap.recording ? .red : .blue)
            Text("\(cap.frames) 枚" + (cap.dropped > 0 ? "（書き込み待ちで \(cap.dropped) 枚落とした）" : "")).font(.callout.monospacedDigit())
            Text(cap.status).font(.caption).foregroundStyle(.secondary)
            Divider()
            Text("保存済み（ファイル App の TamasenScan）").font(.caption.bold())
            List {
                ForEach(sessions, id: \.self) { s in Text(s).font(.caption.monospaced()) }
                    .onDelete { idx in
                        for i in idx { try? FileManager.default.removeItem(at: docs.appendingPathComponent(String(sessions[i].split(separator: " ")[0]))) }
                        refresh()
                    }
            }
            .listStyle(.plain)
        }
    }

    var docs: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    func refresh() {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: docs.path)) ?? []
        sessions = names.filter { $0.hasPrefix("scan-") }.sorted(by: >).map { n in
            let c = (try? FileManager.default.contentsOfDirectory(atPath: docs.appendingPathComponent(n).appendingPathComponent("frames").path).filter { $0.hasSuffix(".jpg") }.count) ?? 0
            return "\(n)  \(c) 枚"
        }
    }
}
