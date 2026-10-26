import AVFoundation
import Foundation

/// 보이는 물건 중 꼭 필요한 것만 골라 한국어 음성으로 위치를 알려준다.
///
/// 한 화면에 물건이 여러 개 잡히므로 전부 말하면 쓸 수가 없다. 그래서
/// - 확신도 70% 이상, 3m 이내, 5프레임 이상 연속으로 보인 물건만 후보로 삼고
/// - 그중 가장 가까운 것 하나만 말한다
/// - 같은 물건은 8초, 같은 종류는 4초 동안 다시 말하지 않는다
/// - 말하는 중에는 새로 말하지 않고, 끝난 뒤에도 1.5초는 쉰다
@MainActor
final class VoiceGuide: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published var isOn = false {
        didSet {
            if isOn { activateAudio() } else { synthesizer.stopSpeaking(at: .immediate) }
        }
    }
    @Published private(set) var lastSentence = ""

    let minConfidence: Float = 0.7
    let maxDistance: Float = 3.0
    let minSeenFrames = 5
    private let sameObjectCooldown: TimeInterval = 8
    private let sameLabelCooldown: TimeInterval = 4
    private let gapAfterSpeech: TimeInterval = 1.5

    private let synthesizer = AVSpeechSynthesizer()
    private let voice = AVSpeechSynthesisVoice(language: "ko-KR")
    private var spokenObjects: [UUID: Date] = [:]
    private var spokenLabels: [String: Date] = [:]
    private var lastSpeechEnd = Date.distantPast

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func update(_ detections: [Detection]) {
        guard isOn, !synthesizer.isSpeaking, Date().timeIntervalSince(lastSpeechEnd) >= gapAfterSpeech else { return }
        let now = Date()
        let candidates = detections.filter { d in
            d.confidence >= minConfidence
                && d.seenFrames >= minSeenFrames
                && (d.distance.map { $0 <= maxDistance } ?? true)  // LiDAR가 없으면 거리 조건은 건너뛴다
                && now.timeIntervalSince(spokenObjects[d.id] ?? .distantPast) >= sameObjectCooldown
                && now.timeIntervalSince(spokenLabels[d.label] ?? .distantPast) >= sameLabelCooldown
        }
        // 가장 가까운 것 하나. 거리를 모르면 확신도가 높은 것
        guard let target = candidates.min(by: { ($0.distance ?? .infinity, -$0.confidence) < ($1.distance ?? .infinity, -$1.confidence) })
        else { return }

        let sentence = describe(target)
        spokenObjects[target.id] = now
        spokenLabels[target.label] = now
        lastSentence = sentence
        let utterance = AVSpeechUtterance(string: sentence)
        utterance.voice = voice
        utterance.rate = 0.55  // 기본(0.5)보다 조금 빠르게
        synthesizer.speak(utterance)
    }

    /// "왼쪽 1.2미터, 의자"처럼 방향·거리·이름 순서로 만든다. 방향은 화면을 세 칸으로 나눠 정한다.
    private func describe(_ d: Detection) -> String {
        let x = d.rect.midX
        let direction = x < 1.0 / 3 ? "왼쪽" : (x > 2.0 / 3 ? "오른쪽" : "앞")
        let name = KoreanLabels.name(for: d.label)
        guard let m = d.distance else { return "\(direction), \(name)" }
        let distance = m < 1 ? "\(Int((m * 100 / 10).rounded()) * 10)센티" : String(format: "%.1f미터", m)
        return "\(direction) \(distance), \(name)"
    }

    /// 무음 모드에서도 들리고, 음악이 나오면 잠깐 소리를 줄이고 말한다.
    private func activateAudio() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.lastSpeechEnd = Date() }
    }
}
