// 深度カメラ（背面 LiDAR は ARKit sceneDepth、前面 TrueDepth は AVFoundation）から RGB と位置合わせ済み深度を同期して受け取り、セッションとして保存する。
// 保存形式は scan/README.md（session.json、frames/NNNNNN.{jpg,depth,json}）。
// バッファは回転・鏡像化しない（センサーの向きのまま）。内部パラメータもその向き。
import ARKit
import AVFoundation
import CoreImage
import UIKit

enum CamKind: String, CaseIterable, Identifiable {
    case lidar, truedepth
    var id: String { rawValue }
    var label: String { self == .lidar ? "背面 LiDAR" : "前面 TrueDepth" }
}

final class Capture: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate, ARSessionDelegate {
    let session = AVCaptureSession()
    // iPad Pro 2020（iPad8,9）は AVFoundation の builtInLiDARDepthCamera が使えないため、背面は ARKit で受ける
    let ar = ARSession()
    @Published var usingAR = false
    @Published var status = "起動中"
    @Published var recording = false
    @Published var frames = 0
    @Published var dropped = 0
    @Published var depthImage: UIImage?
    @Published var centerDepthMM: Float = 0
    @Published var validPct: Float = 0
    @Published var formatText = ""
    @Published private(set) var device: AVCaptureDevice?

    var recordFPS: Double = 10
    var filtering = false
    var lockAEAF = true

    private let videoOut = AVCaptureVideoDataOutput()
    private let depthOut = AVCaptureDepthDataOutput()
    private var sync: AVCaptureDataOutputSynchronizer?
    private let queue = DispatchQueue(label: "capture")
    private let writeQueue = DispatchQueue(label: "write", qos: .userInitiated)
    private let ci = CIContext()
    private var cam: CamKind = .lidar
    private var sessionDir: URL?
    private var lastSaved: Double = -1
    private var lastPreview: Double = -1
    private var inFlight = 0
    private let inFlightLock = NSLock()
    private var counter = 0
    private var t0: Double = 0
    private var dims = (vw: 0, vh: 0, dw: 0, dh: 0)

    override init() {
        super.init()
        ar.delegate = self
        ar.delegateQueue = queue
        for n in [AVCaptureSession.runtimeErrorNotification, AVCaptureSession.wasInterruptedNotification] {
            NotificationCenter.default.addObserver(forName: n, object: session, queue: .main) { [weak self] note in
                NSLog("TSCAN %@ %@", n.rawValue, note.userInfo?.description ?? "")
                self?.status = "カメラ停止: \((note.userInfo?[AVCaptureSessionErrorKey] as? NSError)?.localizedDescription ?? "中断")"
            }
        }
    }

    func configure(_ kind: CamKind) {
        queue.async { self._configure(kind) }
    }

    private func _configure(_ kind: CamKind) {
        NSLog("TSCAN configure %@", kind.rawValue)
        cam = kind
        session.stopRunning()
        session.beginConfiguration()
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        if kind == .lidar {
            session.commitConfiguration()
            guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
                publish { self.status = "この端末は ARKit の sceneDepth が使えません" }
                return
            }
            let cfg = ARWorldTrackingConfiguration()
            cfg.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
            if let f = ARWorldTrackingConfiguration.supportedVideoFormats.max(by: {
                $0.imageResolution.width * $0.imageResolution.height < $1.imageResolution.width * $1.imageResolution.height }) {
                cfg.videoFormat = f
            }
            cfg.isAutoFocusEnabled = true
            ar.run(cfg, options: [.resetTracking])
            let r = cfg.videoFormat.imageResolution
            dims = (Int(r.width), Int(r.height), 256, 192)
            NSLog("TSCAN configured lidar(ARKit) video=%dx%d", Int(r.width), Int(r.height))
            publish { self.device = nil; self.usingAR = true; self.formatText = "ARKit 映像 \(Int(r.width))×\(Int(r.height))  深度 256×192"; self.status = "背面 LiDAR（ARKit）準備完了" }
            return
        }
        ar.pause()
        publish { self.usingAR = false }
        session.sessionPreset = .inputPriority
        let dev: AVCaptureDevice? = kind == .lidar
            ? AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
            : AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front)
        guard let dev, let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            session.commitConfiguration()
            publish { self.status = "\(kind.label) が使えません" }
            return
        }
        session.addInput(input)
        videoOut.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOut.alwaysDiscardsLateVideoFrames = true
        depthOut.alwaysDiscardsLateDepthData = true
        depthOut.isFilteringEnabled = filtering
        session.addOutput(videoOut)
        session.addOutput(depthOut)

        // 深度つきの形式のうち、映像が 1920 幅以下で最大のもの → その中で深度が最大のもの
        func dimsOf(_ f: AVCaptureDevice.Format) -> CMVideoDimensions { CMVideoFormatDescriptionGetDimensions(f.formatDescription) }
        let f32 = { (d: AVCaptureDevice.Format) in CMFormatDescriptionGetMediaSubType(d.formatDescription) == kCVPixelFormatType_DepthFloat32 }
        let cands = dev.formats.filter { f in
            dimsOf(f).width <= 1920 && f.supportedDepthDataFormats.contains(where: f32)
        }
        let best = cands.max { a, b in
            let da = dimsOf(a), db = dimsOf(b)
            let pa = Int(da.width) * Int(da.height), pb = Int(db.width) * Int(db.height)
            if pa != pb { return pa < pb }
            let ma = a.supportedDepthDataFormats.filter(f32).map { Int(dimsOf($0).width) }.max() ?? 0
            let mb = b.supportedDepthDataFormats.filter(f32).map { Int(dimsOf($0).width) }.max() ?? 0
            return ma < mb
        }
        if let best, let dfmt = best.supportedDepthDataFormats.filter(f32).max(by: { dimsOf($0).width < dimsOf($1).width }) {
            do {
                try dev.lockForConfiguration()
                dev.activeFormat = best
                dev.activeDepthDataFormat = dfmt
                if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
                if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
                dev.unlockForConfiguration()
            } catch {}
            let v = dimsOf(best), d = dimsOf(dfmt)
            dims = (Int(v.width), Int(v.height), Int(d.width), Int(d.height))
        }
        if let c = videoOut.connection(with: .video) {
            if c.isCameraIntrinsicMatrixDeliverySupported { c.isCameraIntrinsicMatrixDeliveryEnabled = true }
            if c.isVideoMirroringSupported { c.automaticallyAdjustsVideoMirroring = false; c.isVideoMirrored = false }
        }
        if let c = depthOut.connection(with: .depthData), c.isVideoMirroringSupported {
            c.automaticallyAdjustsVideoMirroring = false; c.isVideoMirrored = false
        }
        sync = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOut, depthOut])
        sync?.setDelegate(self, queue: queue)
        session.commitConfiguration()
        session.startRunning()
        NSLog("TSCAN configured %@ running=%d video=%dx%d depth=%dx%d", kind.rawValue, session.isRunning ? 1 : 0, dims.vw, dims.vh, dims.dw, dims.dh)
        let ft = "映像 \(dims.vw)×\(dims.vh)  深度 \(dims.dw)×\(dims.dh)"
        publish { self.device = dev; self.formatText = ft; self.status = "\(kind.label) 準備完了" }
    }

    func setFiltering(_ on: Bool) {
        filtering = on
        queue.async { self.depthOut.isFilteringEnabled = on }
    }

    // MARK: 記録

    func start() {
        queue.async {
            let df = DateFormatter(); df.dateFormat = "yyyyMMdd-HHmmss"
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let dir = docs.appendingPathComponent("scan-\(df.string(from: Date()))")
            try? FileManager.default.createDirectory(at: dir.appendingPathComponent("frames"), withIntermediateDirectories: true)
            self.sessionDir = dir
            self.counter = 0
            self.lastSaved = -1
            self.t0 = CACurrentMediaTime()
            if self.lockAEAF, let dev = self.device, (try? dev.lockForConfiguration()) != nil {
                if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
                if dev.isExposureModeSupported(.locked) { dev.exposureMode = .locked }
                dev.unlockForConfiguration()
            }
            self.writeSessionJSON()
            self.publish { self.recording = true; self.frames = 0; self.dropped = 0; self.status = "記録中 \(dir.lastPathComponent)" }
        }
    }

    func stop() {
        queue.async {
            let dir = self.sessionDir
            self.sessionDir = nil
            if let dev = self.device, (try? dev.lockForConfiguration()) != nil {
                if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
                if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
                dev.unlockForConfiguration()
            }
            let n = self.counter
            self.writeQueue.async {
                if let dir { self.writeSessionJSON(dir: dir, frames: n) }
                self.publish { self.recording = false; self.status = "保存しました \(dir?.lastPathComponent ?? "")（\(self.counter) 枚）" }
            }
        }
    }

    private func writeSessionJSON(dir: URL? = nil, frames: Int = 0) {
        guard let dir = dir ?? sessionDir else { return }
        let j: [String: Any] = [
            "camera": cam.rawValue, "video_w": dims.vw, "video_h": dims.vh, "depth_w": dims.dw, "depth_h": dims.dh,
            "fps": recordFPS, "device": UIDevice.current.name + " / " + deviceModel(),
            "created": ISO8601DateFormatter().string(from: Date()), "frames": frames,
            "depth_filtered": filtering, "ae_af_locked": lockAEAF && cam == .truedepth, "source": cam == .lidar ? "arkit" : "avfoundation", "app": "TamasenScan 1",
        ]
        if let d = try? JSONSerialization.data(withJSONObject: j, options: [.prettyPrinted, .sortedKeys]) {
            try? d.write(to: dir.appendingPathComponent("session.json"))
        }
    }

    // MARK: 受信

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let vd = collection.synchronizedData(for: videoOut) as? AVCaptureSynchronizedSampleBufferData,
              !vd.sampleBufferWasDropped,
              let dd = collection.synchronizedData(for: depthOut) as? AVCaptureSynchronizedDepthData,
              !dd.depthDataWasDropped,
              let pix = CMSampleBufferGetImageBuffer(vd.sampleBuffer) else { return }
        var depth = dd.depthData
        if depth.depthDataType != kCVPixelFormatType_DepthFloat32 {
            depth = depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        }
        let now = CACurrentMediaTime()
        let dmap = depth.depthDataMap
        let dw = CVPixelBufferGetWidth(dmap), dh = CVPixelBufferGetHeight(dmap)
        let dData = copyDepth(dmap)

        if now - lastPreview > 0.12 {
            lastPreview = now
            let (img, c, pct) = colorize(dData, dw, dh)
            publish { self.depthImage = img; self.centerDepthMM = c; self.validPct = pct }
        }
        guard sessionDir != nil, now - lastSaved >= 1.0 / recordFPS - 0.005 else { return }
        var meta: [String: Any] = [
            "depth_filtered": depth.isDepthDataFiltered,
            "video_w": CVPixelBufferGetWidth(pix), "video_h": CVPixelBufferGetHeight(pix),
            "depth_w": dw, "depth_h": dh,
            "depth_accuracy": depth.depthDataAccuracy == .absolute ? "absolute" : "relative",
            "depth_quality": depth.depthDataQuality == .high ? "high" : "low",
        ]
        if let cal = depth.cameraCalibrationData {
            let m = cal.intrinsicMatrix   // 列優先 m[列][行]
            meta["K"] = [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]].map { Double($0) }
            meta["K_ref"] = [Double(cal.intrinsicMatrixReferenceDimensions.width), Double(cal.intrinsicMatrixReferenceDimensions.height)]
            meta["dist_center"] = [Double(cal.lensDistortionCenter.x), Double(cal.lensDistortionCenter.y)]
            meta["lut"] = floats(cal.lensDistortionLookupTable)
            meta["inv_lut"] = floats(cal.inverseLensDistortionLookupTable)
            meta["pixel_size_mm"] = Double(cal.pixelSize)
        }
        if let a = CMGetAttachment(vd.sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data,
           a.count == MemoryLayout<matrix_float3x3>.size {
            let m = a.withUnsafeBytes { $0.load(as: matrix_float3x3.self) }
            meta["K_video"] = [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]].map { Double($0) }
        }
        if let dev = device { meta["lens_position"] = Double(dev.lensPosition); meta["exposure_s"] = CMTimeGetSeconds(dev.exposureDuration); meta["iso"] = Double(dev.iso) }
        ingest(pix: pix, depth: dData, conf: nil, meta: meta, now: now)
    }

    /// 記録中なら 1 フレームを書き込み待ちに積む（呼び出し側で fps の間引きは済ませる）
    private func ingest(pix: CVPixelBuffer, depth dData: Data, conf: Data?, meta m: [String: Any], now: Double) {
        guard let dir = sessionDir else { return }
        inFlightLock.lock()
        let busy = inFlight >= 4
        if !busy { inFlight += 1 }
        inFlightLock.unlock()
        if busy { publish { self.dropped += 1 }; return }
        lastSaved = now
        let idx = counter
        counter += 1
        var meta = m
        meta["t"] = now - t0
        let ciimg = CIImage(cvPixelBuffer: pix)
        writeQueue.async {
            if let conf { try? conf.write(to: dir.appendingPathComponent("frames").appendingPathComponent(String(format: "%06d.conf", idx))) }
            let base = dir.appendingPathComponent("frames").appendingPathComponent(String(format: "%06d", idx))
            if let cs = CGColorSpace(name: CGColorSpace.sRGB),
               let jpg = self.ci.jpegRepresentation(of: ciimg, colorSpace: cs,
                                                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.92]) {
                try? jpg.write(to: base.appendingPathExtension("jpg"))
            }
            try? dData.write(to: base.appendingPathExtension("depth"))
            if let d = try? JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys]) {
                try? d.write(to: base.appendingPathExtension("json"))
            }
            self.inFlightLock.lock(); self.inFlight -= 1; self.inFlightLock.unlock()
            self.publish { self.frames = idx + 1 }
        }
    }

    // MARK: ARKit（背面 LiDAR）

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let sd = (filtering ? frame.smoothedSceneDepth : nil) ?? frame.sceneDepth else { return }
        let now = CACurrentMediaTime()
        let dw = CVPixelBufferGetWidth(sd.depthMap), dh = CVPixelBufferGetHeight(sd.depthMap)
        let dData = copyDepth(sd.depthMap)
        if now - lastPreview > 0.12 {
            lastPreview = now
            let (img, c, pct) = colorize(dData, dw, dh)
            publish { self.depthImage = img; self.centerDepthMM = c; self.validPct = pct }
        }
        guard sessionDir != nil, now - lastSaved >= 1.0 / recordFPS - 0.005 else { return }
        let conf = sd.confidenceMap.map { copyDepth($0, bpp: 1) }
        let m = frame.camera.intrinsics   // 列優先、capturedImage の解像度での値
        let r = frame.camera.imageResolution
        let T = frame.camera.transform     // world ← camera（ARKit のカメラ座標: x 右, y 上, z 後ろ。センサーの横向き）
        let pix = frame.capturedImage
        var meta: [String: Any] = [
            "source": "arkit", "depth_filtered": filtering && frame.smoothedSceneDepth != nil,
            "video_w": CVPixelBufferGetWidth(pix), "video_h": CVPixelBufferGetHeight(pix),
            "depth_w": dw, "depth_h": dh, "depth_accuracy": "absolute",
            "K": [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]].map { Double($0) },
            "K_ref": [Double(r.width), Double(r.height)], "dist_center": [Double](), "lut": [Double](), "inv_lut": [Double](),
            "T_world_cam_arkit": (0..<4).flatMap { row in (0..<4).map { col in Double(T[col][row]) } },
            "tracking": "\(frame.camera.trackingState)",
            "exposure_s": frame.camera.exposureDuration,
        ]
        if conf != nil { meta["conf"] = "uint8 0=low 1=medium 2=high, depth_w*depth_h" }
        ingest(pix: pix, depth: dData, conf: conf, meta: meta, now: now)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        NSLog("TSCAN ARKit error %@", error.localizedDescription)
        publish { self.status = "ARKit エラー: \(error.localizedDescription)" }
    }

    // MARK: 補助

    private func copyDepth(_ buf: CVPixelBuffer, bpp: Int = 4) -> Data {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        let rb = CVPixelBufferGetBytesPerRow(buf)
        var out = Data(count: w * h * bpp)
        guard let src = CVPixelBufferGetBaseAddress(buf) else { return out }
        out.withUnsafeMutableBytes { dst in
            for y in 0..<h {
                memcpy(dst.baseAddress! + y * w * bpp, src + y * rb, w * bpp)
            }
        }
        return out
    }

    private func colorize(_ d: Data, _ w: Int, _ h: Int) -> (UIImage?, Float, Float) {
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        var valid = 0
        var center: Float = 0
        d.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float32.self)
            center = f[(h / 2) * w + w / 2]
            for i in 0..<(w * h) {
                let z = f[i]
                guard z.isFinite, z > 0 else { continue }
                valid += 1
                let t = max(0, min(1, (z - 0.15) / 0.85))   // 0.15〜1.0 m
                rgba[i * 4] = UInt8(255 * (1 - t)); rgba[i * 4 + 1] = UInt8(255 * (1 - abs(t - 0.5) * 2)); rgba[i * 4 + 2] = UInt8(255 * t); rgba[i * 4 + 3] = 255
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let img: UIImage? = rgba.withUnsafeMutableBytes { p in
            guard let ctx = CGContext(data: p.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                  let cg = ctx.makeImage() else { return nil }
            return UIImage(cgImage: cg)
        }
        return (img, center.isFinite ? center * 1000 : 0, Float(valid) / Float(w * h) * 100)
    }

    private func floats(_ d: Data?) -> [Double] {
        guard let d else { return [] }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)).map { Double($0) } }
    }

    private func deviceModel() -> String {
        var s = utsname(); uname(&s)
        return withUnsafeBytes(of: &s.machine) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
    }

    private func publish(_ f: @escaping () -> Void) { DispatchQueue.main.async(execute: f) }
}
