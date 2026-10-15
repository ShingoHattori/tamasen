// Apple Object Capture（RealityKit PhotogrammetrySession）で、TamasenScan の記録から 3D モデルを作る。
// 入力: セッションの frames/NNNNNN.{jpg,depth} と、prep_masks.py のマスク <masks>/NNNNNN.png（ある画像だけ使う）。
// 深度（m、映像と同じ視野）を渡すので、出力の大きさは実寸（m）。座標系は Object Capture が決める（マーカー座標ではない）。
// ビルド: swiftc -O scan/photogrammetry/objcap.swift -o scan/photogrammetry/objcap
// 使い方: objcap <session> <masks|-> <出力ディレクトリ> [reduced|medium|full|raw] [nodepth]
import CoreImage
import CoreVideo
import Foundation
import RealityKit

let a = CommandLine.arguments
guard a.count >= 4 else { print("usage: objcap <session> <masks> <out> [detail]"); exit(2) }
let session = URL(fileURLWithPath: a[1]), masks = URL(fileURLWithPath: a[2]), out = URL(fileURLWithPath: a[3])
let detail: PhotogrammetrySession.Request.Detail = {
    switch a.count > 4 ? a[4] : "medium" {
    case "reduced": return .reduced
    case "full": return .full
    case "raw": return .raw
    default: return .medium
    }
}()

// 5 番目の引数 nodepth で深度を渡さない（画像とマスクだけで形を作る。尺度は後でマーカーから決める）
let noDepth = a.count > 5 && a[5] == "nodepth"
let ci = CIContext()
func pixelBuffer(_ img: CIImage, _ fmt: OSType) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
    CVPixelBufferCreate(nil, Int(img.extent.width), Int(img.extent.height), fmt, attrs as CFDictionary, &pb)
    guard let pb else { return nil }
    ci.render(img, to: pb)
    return pb
}

func depthBuffer(_ url: URL, _ w: Int, _ h: Int) -> CVPixelBuffer? {
    guard let d = try? Data(contentsOf: url), d.count == w * h * 4 else { return nil }
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_DepthFloat32, nil, &pb)
    guard let pb else { return nil }
    CVPixelBufferLockBaseAddress(pb, [])
    let rb = CVPixelBufferGetBytesPerRow(pb), base = CVPixelBufferGetBaseAddress(pb)!
    d.withUnsafeBytes { src in
        for y in 0..<h {
            let row = base.advanced(by: y * rb).assumingMemoryBound(to: Float32.self)
            for x in 0..<w {
                var v = src.load(fromByteOffset: (y * w + x) * 4, as: Float32.self)
                if !v.isFinite || v <= 0 { v = 0 }
                row[x] = v
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, [])
    return pb
}

// masks に "-" を渡すとマスクなし（全フレーム、Object Capture の自動マスク）
let useMasks = a[2] != "-"
let listDir = useMasks ? masks.path : session.appendingPathComponent("frames").path
let maskNames = ((try? FileManager.default.contentsOfDirectory(atPath: listDir)) ?? [])
    .filter { $0.hasSuffix(useMasks ? ".png" : ".jpg") }.sorted().map { useMasks ? $0 : String($0.dropLast(4)) + ".png" }
var samples: [PhotogrammetrySample] = []
for (i, m) in maskNames.enumerated() {
    let stem = String(m.dropLast(4))
    let f = session.appendingPathComponent("frames").appendingPathComponent(stem)
    guard let meta = try? JSONSerialization.jsonObject(with: Data(contentsOf: f.appendingPathExtension("json"))) as? [String: Any],
          let img = CIImage(contentsOf: f.appendingPathExtension("jpg")),
          let pb = pixelBuffer(img, kCVPixelFormatType_32BGRA) else { continue }
    var s = PhotogrammetrySample(id: Int(stem) ?? i, image: pb)
    if useMasks {
        guard let mimg = CIImage(contentsOf: masks.appendingPathComponent(m)),
              let mb = pixelBuffer(mimg, kCVPixelFormatType_OneComponent8) else { continue }
        s.objectMask = mb
    }
    if !noDepth, let dw = meta["depth_w"] as? Int, let dh = meta["depth_h"] as? Int,
       let db = depthBuffer(f.appendingPathExtension("depth"), dw, dh) {
        s.depthDataMap = db
    }
    samples.append(s)
}
print("samples \(samples.count)")

var cfg = PhotogrammetrySession.Configuration()
cfg.sampleOrdering = .sequential
cfg.featureSensitivity = .high
cfg.isObjectMaskingEnabled = true

let ps = try PhotogrammetrySession(input: samples, configuration: cfg)
let done = DispatchSemaphore(value: 0)
Task {
    for try await o in ps.outputs {
        switch o {
        case .requestProgress(_, let f): print(String(format: "progress %.0f%%", f * 100))
        case .requestComplete(_, .poses(let p)):
            // 各サンプル（id = フレーム番号）のカメラ姿勢（モデル座標、4x4 列優先）を書き出す
            var d: [String: [Float]] = [:]
            for (id, sp) in p.posesBySample {
                let m = sp.transform.matrix
                d[String(id)] = [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
            }
            let url = out.appendingPathComponent("poses.json")
            try? JSONSerialization.data(withJSONObject: d).write(to: url)
            print("poses \(d.count) -> \(url.path)")
        case .requestComplete(_, let r): print("complete \(r)")
        case .requestError(_, let e): print("error \(e)")
        case .processingComplete: done.signal()
        case .invalidSample(let id, let r): print("invalid sample \(id): \(r)")
        case .skippedSample(let id): print("skipped sample \(id)")
        case .automaticDownsampling: print("automatic downsampling")
        case .processingCancelled: print("cancelled"); done.signal()
        default: break
        }
    }
}
try ps.process(requests: [.modelFile(url: out, detail: detail), .poses])
done.wait()
