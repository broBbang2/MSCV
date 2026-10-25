import AVFoundation
import CoreImage
import CoreML
import Vision

/// 화면에 그릴 탐지 결과 하나. rect는 사진 기준 0~1 좌표이고 원점은 왼쪽 위.
struct Detection: Identifiable {
    var id = UUID()  // Tracker가 같은 물건에는 같은 id를 붙인다
    let label: String
    let confidence: Float
    let rect: CGRect
    var distance: Float?  // LiDAR로 잰 거리(m). 깊이를 못 받으면 nil
}

/// 후면 카메라 프레임을 받아 Detector(학습한 YOLO, NMS 포함 CoreML)로 물건을 찾는다.
/// LiDAR가 있으면 깊이 지도를 영상과 같은 순간 것끼리 받아 물건까지의 거리를 잰다.
/// 앱이 켜져 있는 동안 0.5초마다 원본 프레임과 성능 기록을 Documents에 저장한다 (맥이 08_pull_app.py로 가져간다).
final class CameraDetector: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                            AVCaptureDataOutputSynchronizerDelegate {
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var inferenceMs: Double = 0
    @Published var errorMessage: String?
    @Published private(set) var isRecording = true
    @Published private(set) var savedCount = 0
    @Published private(set) var hasDepth = false
    @Published private(set) var isTestMode = false
    /// 카메라 프레임 크기 (세로 화면 기준). 박스를 화면 좌표로 옮길 때 쓴다. 첫 프레임에서 실제 크기로 바뀐다.
    @Published private(set) var frameSize = CGSize(width: 720, height: 1280)

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?

    // 아래 값들은 모두 queue 안에서만 읽고 쓴다
    private let queue = DispatchQueue(label: "camera.frames")
    private var request: VNCoreMLRequest?
    private var lastFrameTime = CACurrentMediaTime()
    private var smoothFps = 0.0
    private var smoothInferenceMs = 0.0
    private var recording = true
    private let tracker = Tracker()
    private var lastFrameSize = CGSize.zero
    private var testMode = false
    // 깊이는 영상의 절반 정도 속도로 와서, 비는 프레임엔 직전 깊이 지도를 다시 쓴다 (짧은 시간이면 거리가 거의 그대로다)
    private var lastDepth: AVDepthData?
    private var lastDepthTime: CFTimeInterval = 0
    private let depthMaxAge: CFTimeInterval = 0.1

    // 저장: 앱을 켤 때마다 새 세션. 파일 이름 앞부분(app날짜-시각)이 02/04 스크립트의 "영상 이름" 역할을 한다
    private let saveInterval: CFTimeInterval = 0.5
    private var sessionName = ""
    private var lastSaveTime: CFTimeInterval = 0
    private var frameIndex = 0
    private var logHandle: FileHandle?
    private let ciContext = CIContext()
    private let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    private var framesDir: URL { documents.appendingPathComponent("frames") }

    /// 모델 안의 NMS가 읽는 임계값. 앱에서 바꾸면 다음 프레임부터 적용된다.
    private final class Thresholds: NSObject, MLFeatureProvider {
        var confidence = 0.25
        var iou = 0.45
        var featureNames: Set<String> { ["confidenceThreshold", "iouThreshold"] }
        func featureValue(for name: String) -> MLFeatureValue? {
            switch name {
            case "confidenceThreshold": MLFeatureValue(double: confidence)
            case "iouThreshold": MLFeatureValue(double: iou)
            default: nil
            }
        }
    }
    private let thresholds = Thresholds()

    func start() {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                self.removePulledFiles()
                self.startSession()
                try self.setUpModel()
                try self.setUpCamera()
                self.session.startRunning()
            } catch {
                DispatchQueue.main.async { self.errorMessage = error.localizedDescription }
            }
        }
    }

    func setConfidence(_ value: Double) {
        queue.async { self.thresholds.confidence = value }
    }

    /// 저장을 잠깐 멈추거나 다시 시작한다 (화면 인식은 계속된다).
    func toggleRecording() {
        queue.async {
            self.recording.toggle()
            let now = self.recording
            DispatchQueue.main.async { self.isRecording = now }
        }
    }

    // MARK: - 저장

    /// 맥이 가져간 파일 목록(pulled.txt)을 읽어 휴대폰에서 지운다. 기기 파일을 직접 지울 방법이 없어서 앱이 대신 지운다.
    private func removePulledFiles() {
        let list = documents.appendingPathComponent("pulled.txt")
        guard let text = try? String(contentsOf: list, encoding: .utf8) else { return }
        for name in text.split(separator: "\n") {
            let file = name.hasSuffix(".csv") ? documents.appendingPathComponent(String(name))
                                              : framesDir.appendingPathComponent(String(name))
            try? FileManager.default.removeItem(at: file)
        }
        try? FileManager.default.removeItem(at: list)
    }

    /// 시험용 모드를 켜고 끈다. 바꿀 때마다 새 묶음(세션)을 시작해서 시험용 사진과 학습용 사진이 섞이지 않게 한다.
    /// 시험용 묶음은 이름이 test_로 시작하고, 맥의 04_auto_label.py가 학습에 넣지 않고 채점에만 쓴다.
    func setTestMode(_ on: Bool) {
        queue.async {
            guard self.testMode != on else { return }
            self.testMode = on
            try? self.logHandle?.close()
            self.startSession()
            DispatchQueue.main.async { self.isTestMode = on }
        }
    }

    private func startSession() {
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyyMMdd-HHmmss"
        sessionName = (testMode ? "test_" : "") + "app\(stamp.string(from: Date()))"
        frameIndex = 0
        try? FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)
        let log = documents.appendingPathComponent("\(sessionName).csv")
        FileManager.default.createFile(atPath: log.path, contents: Data("file,fps,inference_ms,labels\n".utf8))
        logHandle = try? FileHandle(forWritingTo: log)
        _ = try? logHandle?.seekToEnd()
    }

    private func saveFrameIfNeeded(_ pixelBuffer: CVPixelBuffer, at time: CFTimeInterval, detections: [Detection]) {
        guard recording, time - lastSaveTime >= saveInterval else { return }
        lastSaveTime = time
        frameIndex += 1
        let name = String(format: "%@_%06d.jpg", sessionName, frameIndex)
        // LiDAR 카메라 형식은 1440x1920이라, 긴 변을 1280으로 줄여 예전 사진과 크기를 맞추고 용량도 줄인다
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let scale = min(1, 1280 / max(image.extent.width, image.extent.height))
        if scale < 1 { image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) }
        guard let jpeg = ciContext.jpegRepresentation(
            of: image, colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9])
        else { return }
        try? jpeg.write(to: framesDir.appendingPathComponent(name))
        // 이름:확신도[@거리m]
        let labels = detections.map { d in
            "\(d.label):\(Int(d.confidence * 100))" + (d.distance.map { String(format: "@%.2fm", $0) } ?? "")
        }.joined(separator: " ")
        let line = String(format: "%@,%.1f,%.1f,%@\n", name, smoothFps, smoothInferenceMs, labels)
        try? logHandle?.write(contentsOf: Data(line.utf8))
        DispatchQueue.main.async { self.savedCount += 1 }
    }

    // MARK: - 모델과 카메라

    private func setUpModel() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine을 쓸 수 있으면 쓴다
        let vnModel = try VNCoreMLModel(for: Detector(configuration: config).model)
        vnModel.featureProvider = thresholds
        let request = VNCoreMLRequest(model: vnModel)
        // 학습 때와 같은 정사각형 입력으로 늘려 넣는다. Vision이 박스를 원래 사진 좌표로 되돌려준다.
        request.imageCropAndScaleOption = .scaleFill
        self.request = request
    }

    private func setUpCamera() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        let lidar = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
        guard let device = lidar ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw NSError(domain: "firstCV", code: 1, userInfo: [NSLocalizedDescriptionKey: "후면 카메라를 찾을 수 없어요"])
        }
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) { session.addInput(input) }

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true  // 추론이 밀리면 오래된 프레임은 버린다
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        if let lidar, try configureDepth(lidar) {
            // 영상과 깊이를 같은 순간 것끼리 묶어서 받는다
            let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
            sync.setDelegate(self, queue: queue)
            synchronizer = sync
            DispatchQueue.main.async { self.hasDepth = true }
        } else {
            session.sessionPreset = .hd1280x720
            videoOutput.setSampleBufferDelegate(self, queue: queue)
        }
        for output in [videoOutput, depthOutput] as [AVCaptureOutput] {
            if let connection = output.connection(with: output === depthOutput ? .depthData : .video),
               connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90  // 세로로 든 화면과 같은 방향으로 받는다
            }
        }
    }

    /// 깊이를 함께 주는 4:3 영상 형식(가능하면 1920x1440)과 가장 큰 Float32 깊이 형식을 고른다.
    /// 영상과 깊이 지도의 화면비가 같아야 박스 위치로 깊이를 찾을 수 있다.
    private func configureDepth(_ device: AVCaptureDevice) throws -> Bool {
        let formats = device.formats.filter { f in
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return !f.supportedDepthDataFormats.isEmpty && d.width * 3 == d.height * 4 && d.width <= 1920
        }
        guard let format = formats.max(by: {
            CMVideoFormatDescriptionGetDimensions($0.formatDescription).width
                < CMVideoFormatDescriptionGetDimensions($1.formatDescription).width
        }), let depthFormat = format.supportedDepthDataFormats
            .filter({ CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32 })
            .max(by: { CMVideoFormatDescriptionGetDimensions($0.formatDescription).width
                < CMVideoFormatDescriptionGetDimensions($1.formatDescription).width })
        else { return false }

        session.sessionPreset = .inputPriority
        try device.lockForConfiguration()
        device.activeFormat = format
        device.activeDepthDataFormat = depthFormat
        device.unlockForConfiguration()

        depthOutput.isFilteringEnabled = true  // 반사·경계 때문에 비는 곳을 메워준다
        depthOutput.alwaysDiscardsLateDepthData = true
        guard session.canAddOutput(depthOutput) else { return false }
        session.addOutput(depthOutput)
        return true
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let video = collection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              !video.sampleBufferWasDropped else { return }
        let now = CACurrentMediaTime()
        if let synced = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
           !synced.depthDataWasDropped {
            lastDepth = synced.depthData
            lastDepthTime = now
        }
        process(video.sampleBuffer, depth: now - lastDepthTime <= depthMaxAge ? lastDepth : nil)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        process(sampleBuffer, depth: nil)
    }

    /// 박스 가운데 40% 영역의 깊이 중앙값(m). 가장자리는 뒤쪽 배경이 섞여서 뺀다.
    private func distance(of rect: CGRect, in depthData: AVDepthData) -> Float? {
        let depth = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = depth.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let portrait = h > w  // 깊이 지도도 세로로 돌아왔는지. 아니면 센서 원래 방향(가로)이다

        let inner = rect.insetBy(dx: rect.width * 0.3, dy: rect.height * 0.3)
        var values: [Float] = []
        let steps = 8
        for i in 0..<steps {
            for j in 0..<steps {
                let px = inner.minX + inner.width * (CGFloat(i) + 0.5) / CGFloat(steps)
                let py = inner.minY + inner.height * (CGFloat(j) + 0.5) / CGFloat(steps)
                // 세로 화면 (x, y)는 시계방향 90도 돌리기 전 가로 센서에서 (y, 1 - x)
                let (sx, sy) = portrait ? (px, py) : (py, 1 - px)
                let x = min(max(Int(sx * CGFloat(w)), 0), w - 1)
                let y = min(max(Int(sy * CGFloat(h)), 0), h - 1)
                let v = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)[x]
                if v.isFinite && v > 0 { values.append(v) }
            }
        }
        guard values.count >= steps else { return nil }
        values.sort()
        return values[values.count / 2]
    }

    private func process(_ sampleBuffer: CMSampleBuffer, depth: AVDepthData?) {
        guard let request, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        if size != lastFrameSize {
            lastFrameSize = size
            DispatchQueue.main.async { self.frameSize = size }
        }
        let start = CACurrentMediaTime()
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
        let end = CACurrentMediaTime()

        let results = (request.results as? [VNRecognizedObjectObservation]) ?? []
        let detections = results.compactMap { obs -> Detection? in
            guard let top = obs.labels.first else { return nil }
            let b = obs.boundingBox  // Vision은 왼쪽 아래가 원점이라 위아래를 뒤집는다
            let rect = CGRect(x: b.minX, y: 1 - b.maxY, width: b.width, height: b.height)
            return Detection(label: top.identifier, confidence: top.confidence, rect: rect,
                             distance: depth.flatMap { distance(of: rect, in: $0) })
        }
        // 숫자가 덜 흔들리게 지수 평균
        smoothFps = smoothFps * 0.9 + 1 / max(end - lastFrameTime, 1e-3) * 0.1
        smoothInferenceMs = smoothInferenceMs * 0.9 + (end - start) * 1000 * 0.1
        lastFrameTime = end
        saveFrameIfNeeded(pixelBuffer, at: end, detections: detections)  // 기록은 추적 전 원래 결과로
        let shown = tracker.update(detections)

        let fps = smoothFps, inferenceMs = smoothInferenceMs
        DispatchQueue.main.async {
            self.detections = shown
            self.fps = fps
            self.inferenceMs = inferenceMs
        }
    }
}
