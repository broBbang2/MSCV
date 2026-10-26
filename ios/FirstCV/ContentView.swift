import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var detector = CameraDetector()
    @StateObject private var voice = VoiceGuide()
    @State private var confidence = 0.25

    var body: some View {
        GeometryReader { geo in
            ZStack {
                CameraPreview(session: detector.session)
                ForEach(detector.detections) { det in
                    box(for: det, in: geo.size)
                }
            }
        }
        .ignoresSafeArea()
        .overlay(alignment: .top) { hud }
        .overlay(alignment: .bottom) { controls }
        .onAppear { detector.start() }
        .onReceive(detector.$detections) { voice.update($0) }  // 매 프레임 넘겨주고, 말할지는 VoiceGuide가 정한다
    }

    /// 미리보기는 화면을 꽉 채우도록(aspect fill) 잘려서 보이므로, 박스도 같은 방식으로 옮긴다.
    private func box(for det: Detection, in view: CGSize) -> some View {
        let frame = detector.frameSize
        let scale = max(view.width / frame.width, view.height / frame.height)
        let dx = (view.width - frame.width * scale) / 2
        let dy = (view.height - frame.height * scale) / 2
        let rect = CGRect(x: det.rect.minX * frame.width * scale + dx,
                          y: det.rect.minY * frame.height * scale + dy,
                          width: det.rect.width * frame.width * scale,
                          height: det.rect.height * frame.height * scale)
        // 같은 물건은 실행할 때마다 같은 색 (String.hashValue는 실행마다 달라져서 글자 코드 합을 쓴다)
        let hue = Double(det.label.unicodeScalars.reduce(0) { $0 + Int($1.value) * 37 } % 360) / 360
        let color = Color(hue: hue, saturation: 0.8, brightness: 1)
        return ZStack(alignment: .topLeading) {
            Rectangle().stroke(color, lineWidth: 3)
            Text("\(KoreanLabels.name(for: det.label)) \(Int(det.confidence * 100))%"
                 + (det.distance.map { String(format: " · %.1fm", $0) } ?? ""))
                .font(.caption.bold())
                .padding(.horizontal, 4)
                .background(color)
                .foregroundStyle(.black)
                .offset(y: -18)
        }
        .frame(width: rect.width, height: rect.height)
        .position(x: rect.midX, y: rect.midY)
    }

    private var hud: some View {
        HStack(spacing: 16) {
            Text(String(format: "%.0f FPS", detector.fps))
            Text(String(format: "추론 %.1f ms", detector.inferenceMs))
            Text("물건 \(detector.detections.count)개")
            if detector.hasDepth { Label("LiDAR", systemImage: "ruler") }
        }
        .font(.callout.monospacedDigit().bold())
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.black.opacity(0.6), in: Capsule())
        .foregroundStyle(.white)
        .padding(.top, 8)
    }

    private var controls: some View {
        VStack(spacing: 6) {
            if let error = detector.errorMessage {
                Text(error).foregroundStyle(.red)
            }
            Toggle(isOn: $voice.isOn) {
                Text(voice.isOn ? "음성 안내 켜짐 (확신도 70%↑, 3m 이내)" : "음성 안내")
            }
            .tint(.green)
            if voice.isOn, !voice.lastSentence.isEmpty {
                Text("🔊 \(voice.lastSentence)").font(.callout.bold())
            }
            Toggle(isOn: Binding(get: { detector.isTestMode }, set: { detector.setTestMode($0) })) {
                Text(detector.isTestMode ? "시험용으로 저장 중 (학습에 안 씀)" : "시험용 모드")
                    .foregroundStyle(detector.isTestMode ? .orange : .white)
            }
            .tint(.orange)
            HStack {
                Label(detector.isRecording ? "저장 중 \(detector.savedCount)장" : "저장 멈춤 \(detector.savedCount)장",
                      systemImage: detector.isRecording ? "record.circle" : "pause.circle")
                    .foregroundStyle(detector.isRecording ? (detector.isTestMode ? .orange : .red) : .white)
                Spacer()
                Button(detector.isRecording ? "멈춤" : "다시 저장") { detector.toggleRecording() }
                    .buttonStyle(.bordered)
                    .tint(.white)
            }
            Text("확신도 기준 \(Int(confidence * 100))%")
            Slider(value: $confidence, in: 0.05...0.9) { _ in detector.setConfidence(confidence) }
        }
        .font(.footnote.bold())
        .foregroundStyle(.white)
        .padding(14)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal, 16).padding(.bottom, 24)
    }
}

/// AVCaptureSession 화면을 SwiftUI에 붙이는 뷰.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
