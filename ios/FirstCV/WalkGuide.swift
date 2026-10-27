import Foundation
import UIKit

/// 보행 모드: 걷는 길의 장애물·낙차와 다가오는 물체를 진동과 짧은 음성으로 알린다.
///
/// 장애물(솟은 것): 2m 주의(화면만), 1.2m 경고(약한 진동, "앞 1.1미터 의자"), 0.6m 멈춤(강한 진동, "멈추세요")
/// 낙차(꺼진 곳): 넘어지면 더 위험해서 3m 주의, 2m 경고, 1m 멈춤. "앞 1.5미터 내려가는 계단"
/// 다가오는 사람·자전거·차·오토바이: 내 걷는 속도를 빼고도 1초에 0.7m 넘게 다가오고 3초 안에 닿으면
///   "오른쪽에서 자전거 다가와요". 같은 물체는 5초에 한 번만.
/// 거리는 프레임마다 흔들리므로 섞어서 쓰고, 단계가 내려갈 때는 10cm 여유를 둔다 (경계에서 왔다 갔다 하지 않게).
@MainActor
final class WalkGuide: ObservableObject {
    enum Level: Int, Comparable {
        case clear, caution, warn, stop
        static func < (a: Level, b: Level) -> Bool { a.rawValue < b.rawValue }
    }

    /// 장애물이나 낙차 하나의 거리와 단계를 프레임 사이에 이어서 관리한다.
    private struct Hazard {
        let caution: Float, warn: Float, stop: Float
        var distance: Float?
        var lastSeen = Date.distantPast
        var level = Level.clear

        mutating func update(_ found: Float?, now: Date, lostAfter: TimeInterval, hysteresis: Float) {
            if let found {
                distance = distance.map { $0 * 0.5 + found * 0.5 } ?? found
                lastSeen = now
            } else if now.timeIntervalSince(lastSeen) > lostAfter {
                distance = nil
            }
            guard let d = distance else { level = .clear; return }
            let raw: Level = d < stop ? .stop : d < warn ? .warn : d < caution ? .caution : .clear
            guard raw < level else { level = raw; return }
            // 내려갈 때만 여유를 둔다: 지금 단계의 경계보다 10cm 더 멀어져야 한 단계 내려간다
            let bound = level == .stop ? stop : level == .warn ? warn : caution
            if d >= bound + hysteresis { level = raw }
        }
    }

    @Published var isOn = false {
        didSet {
            if isOn {
                light.prepare(); heavy.prepare()
            } else {
                obstacleState.distance = nil; dropState.distance = nil
                level = .clear; obstacle = nil; drop = nil; approaching = nil
            }
        }
    }
    @Published private(set) var level = Level.clear
    @Published private(set) var obstacle: Obstacle?
    @Published private(set) var drop: Drop?
    @Published private(set) var approaching: Detection?

    private var obstacleState = Hazard(caution: 2.0, warn: 1.2, stop: 0.6)
    private var dropState = Hazard(caution: 3.0, warn: 2.0, stop: 1.0)
    private let hysteresis: Float = 0.1
    private let repeatSpeech: TimeInterval = 4
    private let lostAfter: TimeInterval = 0.3  // 이만큼 안 보이면 없어졌다고 본다

    // 다가오는 물체
    private let movers: Set<String> = ["person", "bicycle", "car", "motorcycle"]
    /// 스스로 움직이지 않는 물건. 이것들이 가까워지는 속도 = 내가 걷는 속도
    private let fixed: Set<String> = ["chair", "table", "door", "pole", "bench", "trash can", "traffic cone", "sofa", "stairs"]
    private let assumedWalkSpeed: Float = 1.4  // 고정된 물건이 안 보일 때 걷는 중이면 이 속도로 어림한다
    private let minOwnSpeed: Float = 0.7
    private let maxTimeToReach: Float = 3
    private let maxApproachDistance: Float = 6
    private let approachCooldown: TimeInterval = 5
    private var announced: [UUID: Date] = [:]

    private let voice: VoiceGuide
    private let light = UIImpactFeedbackGenerator(style: .light)
    private let heavy = UIImpactFeedbackGenerator(style: .heavy)
    private var lastBuzz = Date.distantPast
    private var lastSpeech = Date.distantPast

    init(voice: VoiceGuide) {
        self.voice = voice
    }

    func update(_ report: PathReport?, detections: [Detection]) {
        guard isOn else { return }
        let now = Date()
        if let o = report?.obstacle { obstacle = o }
        if let d = report?.drop { drop = d }
        obstacleState.update(report?.obstacle?.distance, now: now, lostAfter: lostAfter, hysteresis: hysteresis)
        dropState.update(report?.drop?.distance, now: now, lostAfter: lostAfter, hysteresis: hysteresis)
        if obstacleState.distance == nil { obstacle = nil }
        if dropState.distance == nil { drop = nil }

        let previous = level
        level = max(obstacleState.level, dropState.level)
        if level != previous { lastBuzz = .distantPast }
        switch level {
        case .stop: buzz(heavy, every: 0.3, now: now)
        case .warn: buzz(light, every: 0.7, now: now)
        case .clear, .caution: break
        }

        if level >= .warn, level > previous || now.timeIntervalSince(lastSpeech) >= repeatSpeech {
            // 같은 단계면 낙차를 먼저 말한다
            let sentence = dropState.level >= obstacleState.level ? drop.map(describe) : obstacle.map(describe)
            if let sentence {
                voice.warn(level == .stop ? "멈추세요, " + sentence : sentence)
                lastSpeech = now
            }
        } else if level < .stop, let mover = approachingMover(in: detections, walking: report?.walking ?? false, now: now) {
            // 멈춰야 할 만큼 가까운 장애물이 없을 때만 다가오는 물체를 말한다
            approaching = mover
            announced[mover.id] = now
            heavy.impactOccurred()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { self.heavy.impactOccurred() }  // 두 번 = 다가오는 것
            voice.warn("\(direction(of: mover))에서 \(KoreanLabels.name(for: mover.label)) 다가와요")
            lastSpeech = now
        }
        if let a = approaching, !detections.contains(where: { $0.id == a.id }) { approaching = nil }
    }

    /// 내 걷는 속도를 빼고도 빠르게 다가와 곧 닿을 사람·탈것 중 가장 먼저 닿을 것.
    private func approachingMover(in detections: [Detection], walking: Bool, now: Date) -> Detection? {
        let reference = detections.filter { fixed.contains($0.label) && $0.seenFrames >= 5 && $0.distance != nil }
            .compactMap(\.approachSpeed).sorted()
        let mySpeed = reference.isEmpty ? (walking ? assumedWalkSpeed : 0) : reference[reference.count / 2]
        return detections.filter { d in
            guard movers.contains(d.label), d.seenFrames >= 5, let dist = d.distance, let speed = d.approachSpeed,
                  dist <= maxApproachDistance, speed > 0 else { return false }
            return speed - mySpeed >= minOwnSpeed && dist / speed <= maxTimeToReach
                && now.timeIntervalSince(announced[d.id] ?? .distantPast) >= approachCooldown
        }.min { $0.distance! / $0.approachSpeed! < $1.distance! / $1.approachSpeed! }
    }

    private func buzz(_ generator: UIImpactFeedbackGenerator, every interval: TimeInterval, now: Date) {
        guard now.timeIntervalSince(lastBuzz) >= interval else { return }
        generator.impactOccurred()
        generator.prepare()
        lastBuzz = now
    }

    private func direction(of d: Detection) -> String {
        d.rect.midX < 1.0 / 3 ? "왼쪽" : (d.rect.midX > 2.0 / 3 ? "오른쪽" : "앞")
    }

    /// "앞 1.1미터 장애물, 오른쪽이 비었어요"
    private func describe(_ o: Obstacle) -> String {
        let d = obstacleState.distance ?? o.distance
        let name = o.label.map { KoreanLabels.name(for: $0) } ?? "장애물"
        var text = "앞 \(VoiceGuide.spoken(distance: d)) \(name)"
        // 옆 칸이 지금 장애물보다 1m 이상 더 멀리까지 비어 있으면 그쪽을 알려준다
        if max(o.leftClear, o.rightClear) >= d + 1 {
            text += o.rightClear >= o.leftClear ? ", 오른쪽이 비었어요" : ", 왼쪽이 비었어요"
        }
        return text
    }

    /// "앞 1.5미터 내려가는 계단" / "앞 90센티 내려가는 턱"
    private func describe(_ drop: Drop) -> String {
        let d = dropState.distance ?? drop.distance
        return "앞 \(VoiceGuide.spoken(distance: d)) 내려가는 " + (drop.label == "stairs" ? "계단" : "턱")
    }

    /// 화면 표시용 한 줄: 낙차·장애물 중 더 급한 것
    var statusText: String? {
        if let drop, dropState.level >= .caution, dropState.level >= obstacleState.level {
            return String(format: "앞 %.1fm · 내려가는 ", dropState.distance ?? drop.distance) + (drop.label == "stairs" ? "계단" : "턱")
        }
        if let obstacle, obstacleState.level >= .caution {
            return String(format: "앞 %.1fm · ", obstacleState.distance ?? obstacle.distance)
                + (obstacle.label.map { KoreanLabels.name(for: $0) } ?? "장애물")
        }
        return nil
    }

    /// 화면에 동그라미로 표시할 위치 (statusText와 같은 것)
    var markedPoint: CGPoint? {
        if let drop, dropState.level >= .caution, dropState.level >= obstacleState.level { return drop.point }
        if let obstacle, obstacleState.level >= .caution { return obstacle.point }
        return nil
    }
}
