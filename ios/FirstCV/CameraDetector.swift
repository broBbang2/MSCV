import AVFoundation
import CoreImage
import CoreML
import CoreMotion

/// 화면에 그릴 탐지 결과 하나. rect는 사진 기준 0~1 좌표이고 원점은 왼쪽 위.
struct Detection: Identifiable {
    var id = UUID()  // Tracker가 같은 물건에는 같은 id를 붙인다
    let label: String
    let confidence: Float
    let rect: CGRect
    var distance: Float?  // LiDAR로 잰 거리(m). 깊이를 못 받으면 nil
    var seenFrames = 1  // Tracker가 이 물건을 몇 프레임째 보고 있는지 (음성 안내는 오래 보인 것만 말한다)
    var approachSpeed: Float?  // 1초에 몇 m씩 가까워지는지 (멀어지면 음수). Tracker가 거리 변화로 잰다
}

/// 후면 카메라 프레임을 받아 Detector(학습한 YOLO, NMS 포함 CoreML)로 물건을 찾는다.
/// Vision을 거치지 않고 모델을 직접 부른다. Vision은 확신도를 클래스끼리의 비율로 바꿔서(44%가 99%로) 넘겨주기 때문이다.
/// LiDAR가 있으면 깊이 지도를 영상과 같은 순간 것끼리 받아 물건까지의 거리를 잰다.
/// 보행 모드에서는 깊이 지도와 중력 방향으로 걷는 길의 장애물을 찾는다 (ObstacleDetector).
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
    @Published private(set) var walkReport: PathReport?
    /// 카메라 프레임 크기 (세로 화면 기준). 박스를 화면 좌표로 옮길 때 쓴다. 첫 프레임에서 실제 크기로 바뀐다.
    @Published private(set) var frameSize = CGSize(width: 720, height: 1280)

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?

    // 아래 값들은 모두 queue 안에서만 읽고 쓴다
    private let queue = DispatchQueue(label: "camera.frames")
    private var model: MLModel?
    private var classLabels: [String] = []
    private var modelInput: CVPixelBuffer?  // 모델 입력 크기(640x640)로 늘린 프레임. 매 프레임 같은 버퍼를 다시 쓴다
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
    // 보행 모드
    private var walkMode = false
    private let obstacles = ObstacleDetector()
    private let motion = CMMotionManager()
    private var shake: Double = 0  // 휴대폰 흔들림(중력 뺀 가속도, g)의 지수 평균. 걷는 중인지 어림한다
    private let walkingShake = 0.05
    private var fovTan: Float = 0  // 사진 긴 변 화각 절반의 tan. LiDAR 형식을 고를 때 정해진다

    // 저장: 앱을 켤 때마다 새 세션. 파일 이름 앞부분(app날짜-시각)이 02/04 스크립트의 "영상 이름" 역할을 한다
    private let saveInterval: CFTimeInterval = 0.5
    private var sessionName = ""
    private var lastSaveTime: CFTimeInterval = 0
    private var frameIndex = 0
    private var logHandle: FileHandle?
    private let ciContext = CIContext()
    private let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    private var framesDir: URL { documents.appendingPathComponent("frames") }

    /// 새 물건으로 보여줄 최소 확신도 (화면 슬라이더). 이미 보이던 물건은 이보다 낮아도 Tracker가 조금 더 붙잡는다
    private var confidence = 0.25
    private let iou = 0.45

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
        queue.async { self.confidence = value }
    }

    /// 저장을 잠깐 멈추거나 다시 시작한다 (화면 인식은 계속된다).
    func toggleRecording() {
        queue.async {
            self.recording.toggle()
            let now = self.recording
            DispatchQueue.main.async { self.isRecording = now }
        }
    }

    /// 보행 모드를 켜고 끈다. 켜져 있는 동안만 중력 센서를 읽는다.
    func setWalkMode(_ on: Bool) {
        queue.async {
            self.walkMode = on
            if on {
                self.motion.deviceMotionUpdateInterval = 1.0 / 60
                self.motion.startDeviceMotionUpdates()
            } else {
                self.motion.stopDeviceMotionUpdates()
                DispatchQueue.main.async { self.walkReport = nil }
            }
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
        let model = try Detector(configuration: config).model
        // 내보낼 때 클래스 이름이 80개로 채워진다 (30번부터는 "30", "31" 같은 빈칸이고 점수는 늘 0)
        classLabels = model.modelDescription.classLabels as? [String] ?? []
        guard let image = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw NSError(domain: "firstCV", code: 2, userInfo: [NSLocalizedDescriptionKey: "모델 입력 형식을 읽을 수 없어요"])
        }
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, image.pixelsWide, image.pixelsHigh, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        modelInput = buffer
        self.model = model
    }

    /// 프레임을 모델 입력 크기의 정사각형으로 늘려 넣고 결과를 받는다.
    /// 비율을 유지하지 않고 늘리지만 정확도 차이는 작고(검증 mAP50 0.284 → 0.277), 박스 좌표를 그대로 쓸 수 있다.
    private func detect(_ pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let model, let input = modelInput else { return [] }
        let frame = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(CVPixelBufferGetWidth(input)) / frame.extent.width
        let sy = CGFloat(CVPixelBufferGetHeight(input)) / frame.extent.height
        ciContext.render(frame.transformed(by: CGAffineTransform(scaleX: sx, y: sy)), to: input)

        // 모델 안 NMS는 Tracker가 붙잡을 수 있게 슬라이더보다 낮은 점수까지 넘겨준다
        guard let features = try? MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: input),
            "confidenceThreshold": MLFeatureValue(double: confidence * Tracker.keepRatio),
            "iouThreshold": MLFeatureValue(double: iou),
        ]), let out = try? model.prediction(from: features),
              let scores = out.featureValue(for: "confidence")?.multiArrayValue,
              let coords = out.featureValue(for: "coordinates")?.multiArrayValue
        else { return [] }

        let count = scores.shape[0].intValue, classes = scores.shape[1].intValue
        let rowStride = scores.strides[0].intValue, coordStride = coords.strides[0].intValue
        let s = scores.dataPointer.assumingMemoryBound(to: Float32.self)
        let c = coords.dataPointer.assumingMemoryBound(to: Float32.self)
        var detections: [Detection] = []
        for i in 0..<count {
            var best = 0
            for k in 1..<classes where s[i * rowStride + k] > s[i * rowStride + best] { best = k }
            guard best < classLabels.count else { continue }
            // 좌표는 0~1의 (가운데 x, 가운데 y, 너비, 높이)이고 원점은 왼쪽 위
            let (x, y, w, h) = (CGFloat(c[i * coordStride]), CGFloat(c[i * coordStride + 1]),
                                CGFloat(c[i * coordStride + 2]), CGFloat(c[i * coordStride + 3]))
            detections.append(Detection(label: classLabels[best], confidence: s[i * rowStride + best],
                                        rect: CGRect(x: x - w / 2, y: y - h / 2, width: w, height: h)))
        }
        return detections
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

        fovTan = Float(tan(Double(format.videoFieldOfView) / 2 * .pi / 180))  // 가로 센서 기준이라 긴 변 화각이다
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

    /// 깊이 지도를 잠근 채로 body를 부른다. body에는 세로 사진 좌표(0~1)의 깊이(m)를 읽는 함수를 넘긴다.
    private func withDepthMap<T>(_ depthData: AVDepthData, _ body: (_ sample: (CGFloat, CGFloat) -> Float?) -> T) -> T? {
        let depth = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = depth.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let portrait = h > w  // 깊이 지도도 세로로 돌아왔는지. 아니면 센서 원래 방향(가로)이다
        return body { px, py in
            // 세로 화면 (x, y)는 시계방향 90도 돌리기 전 가로 센서에서 (y, 1 - x)
            let (sx, sy) = portrait ? (px, py) : (py, 1 - px)
            let x = min(max(Int(sx * CGFloat(w)), 0), w - 1)
            let y = min(max(Int(sy * CGFloat(h)), 0), h - 1)
            let v = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)[x]
            return v.isFinite && v > 0 ? v : nil
        }
    }

    /// 박스 가운데 40% 영역의 깊이 중앙값(m). 가장자리는 뒤쪽 배경이 섞여서 뺀다.
    private func distance(of rect: CGRect, in depthData: AVDepthData) -> Float? {
        withDepthMap(depthData) { sample -> Float? in
            let inner = rect.insetBy(dx: rect.width * 0.3, dy: rect.height * 0.3)
            var values: [Float] = []
            let steps = 8
            for i in 0..<steps {
                for j in 0..<steps {
                    let px = inner.minX + inner.width * (CGFloat(i) + 0.5) / CGFloat(steps)
                    let py = inner.minY + inner.height * (CGFloat(j) + 0.5) / CGFloat(steps)
                    if let v = sample(px, py) { values.append(v) }
                }
            }
            guard values.count >= steps else { return nil }
            values.sort()
            return values[values.count / 2]
        } ?? nil
    }

    /// 걷는 길의 장애물과 낙차. 그 위치를 덮는 탐지 박스 중 가장 작은 것의 이름을 붙인다 (큰 박스는 뒤쪽 배경일 수 있다).
    private func scanPath(in depthData: AVDepthData, size: CGSize, shown: [Detection]) -> PathReport? {
        // 기기 좌표(오른쪽 x, 위 y, 화면 밖 z) → 후면 카메라 좌표(오른쪽 x, 아래 y, 앞 z)
        guard fovTan > 0, let m = motion.deviceMotion else { return nil }
        let a = m.userAcceleration
        shake = shake * 0.95 + (a.x * a.x + a.y * a.y + a.z * a.z).squareRoot() * 0.05
        let gravity = SIMD3<Float>(Float(m.gravity.x), Float(-m.gravity.y), Float(-m.gravity.z))
        let found = withDepthMap(depthData) { sample in
            obstacles.update(gravity: gravity, fovTan: fovTan, aspect: Float(size.width / size.height), sample: sample)
        } ?? nil
        guard var report = found else { return nil }
        func label(at point: CGPoint) -> String? {
            shown.filter { $0.rect.contains(point) }
                .min(by: { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height })?.label
        }
        if let o = report.obstacle { report.obstacle?.label = label(at: o.point) }
        if let d = report.drop { report.drop?.label = label(at: d.point) }
        report.walking = shake > walkingShake
        return report
    }

    private func process(_ sampleBuffer: CMSampleBuffer, depth: AVDepthData?) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        if size != lastFrameSize {
            lastFrameSize = size
            DispatchQueue.main.async { self.frameSize = size }
        }
        let start = CACurrentMediaTime()
        var detections = detect(pixelBuffer)
        let end = CACurrentMediaTime()
        if let depth {
            for i in detections.indices { detections[i].distance = distance(of: detections[i].rect, in: depth) }
        }
        // 숫자가 덜 흔들리게 지수 평균
        smoothFps = smoothFps * 0.9 + 1 / max(end - lastFrameTime, 1e-3) * 0.1
        smoothInferenceMs = smoothInferenceMs * 0.9 + (end - start) * 1000 * 0.1
        lastFrameTime = end
        // 기록은 추적 전 원래 결과로. 슬라이더 기준을 넘은 것만 남겨 예전 기록과 같은 의미로 둔다
        saveFrameIfNeeded(pixelBuffer, at: end, detections: detections.filter { Double($0.confidence) >= confidence })
        let shown = tracker.update(detections, minConfidence: Float(confidence), at: end)
        let report = walkMode ? depth.flatMap { scanPath(in: $0, size: size, shown: shown) } : nil

        let fps = smoothFps, inferenceMs = smoothInferenceMs
        let walking = walkMode
        DispatchQueue.main.async {
            self.detections = shown
            if walking { self.walkReport = report }  // detections 다음에 넣어야 WalkGuide가 이번 프레임 박스를 본다
            self.fps = fps
            self.inferenceMs = inferenceMs
        }
    }
}
