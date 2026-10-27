import CoreGraphics
import Foundation

/// 프레임마다 따로 나오는 탐지 결과를 이어 붙여 화면에서 박스가 깜빡이거나 튀지 않게 한다.
///
/// - 새 물건은 슬라이더 기준(minConfidence) 이상으로 2프레임 연속 잡혀야 보여준다 (한 프레임짜리 오탐을 거른다)
/// - 이미 보이던 물건은 그 60%까지 점수가 내려가도 계속 붙잡는다 (기준선 근처에서 깜빡이지 않게)
/// - 놓쳐도 0.5초까지는 움직이던 방향으로 박스를 옮기며 남겨둔다 (걸을 때 흔들림으로 잠깐 놓치는 경우)
/// - LiDAR 거리가 있으면 1초에 몇 m씩 가까워지는지(approachSpeed)도 잰다. 다가오는 사람·자전거 경고에 쓴다
/// - 많이 겹치면 이름이 달라도 같은 물건으로 보고, 지금까지 쌓인 점수가 가장 큰 이름을 쓴다 (의자↔소파 깜빡임 방지)
final class Tracker {
    /// 보이던 물건을 붙잡는 점수 = 슬라이더 기준 × keepRatio. 모델 안 NMS도 이 점수까지 넘겨준다
    static let keepRatio = 0.6

    private struct Track {
        let id: UUID
        var rect: CGRect
        var velocity: CGVector  // 박스 가운데가 1초에 움직이는 양 (사진 좌표)
        var confidence: Float
        var labelScores: [String: Float]
        var distance: Float?
        var anchor: (distance: Float, time: CFTimeInterval)?  // 가까워지는 속도를 재기 시작한 거리와 시각
        var approach: Float?
        var hits: Int
        var lastSeen: CFTimeInterval
        var shown: Bool

        var label: String { labelScores.max { $0.value < $1.value }!.key }
    }

    private var tracks: [Track] = []
    private var lastTime: CFTimeInterval?
    private let minHits = 2
    private let maxAge: CFTimeInterval = 0.5
    private let minIoU: CGFloat = 0.3
    private let relabelIoU: CGFloat = 0.6  // 이름이 다르면 이만큼은 겹쳐야 같은 물건으로 본다
    private let speedInterval: CFTimeInterval = 0.25  // 거리가 흔들려서 이 정도 간격을 두고 비교한다
    private let smoothing: CGFloat = 0.5  // 새 박스를 얼마나 반영할지 (1이면 그대로, 작을수록 부드럽고 느리다)

    func update(_ detections: [Detection], minConfidence: Float, at time: CFTimeInterval) -> [Detection] {
        let dt = CGFloat(min(time - (lastTime ?? time), 0.2))
        lastTime = time
        let keep = minConfidence * Float(Self.keepRatio)
        var used = Set<Int>()
        var next: [Track] = []

        // 점수 높은 트랙부터, 움직임을 반영해 옮긴 위치와 가장 많이 겹치는 새 박스와 짝짓는다
        for var track in tracks.sorted(by: { $0.confidence > $1.confidence }) {
            let predicted = track.rect.offsetBy(dx: track.velocity.dx * dt, dy: track.velocity.dy * dt)
            var best: Int?
            var bestScore: CGFloat = 0
            for (i, det) in detections.enumerated() where !used.contains(i) && det.confidence >= keep {
                let v = iou(predicted, det.rect)
                let needed = det.label == track.label ? minIoU : relabelIoU
                if v >= needed && v > bestScore { bestScore = v; best = i }
            }
            if let best {
                let det = detections[best]
                used.insert(best)
                let moved = mix(track.rect, det.rect)
                if dt > 0 {
                    let v = CGVector(dx: (moved.midX - track.rect.midX) / dt, dy: (moved.midY - track.rect.midY) / dt)
                    track.velocity = CGVector(dx: track.velocity.dx * 0.5 + v.dx * 0.5, dy: track.velocity.dy * 0.5 + v.dy * 0.5)
                }
                track.rect = moved
                track.confidence = Float(smoothing) * det.confidence + Float(1 - smoothing) * track.confidence
                // 이름 점수는 조금씩 잊으면서 쌓는다. 진짜로 다른 물건이 됐으면 몇 프레임 뒤 이름이 바뀐다
                for key in track.labelScores.keys { track.labelScores[key]! *= 0.9 }
                track.labelScores[det.label, default: 0] += det.confidence
                // 거리도 박스처럼 섞어서 숫자가 덜 흔들리게 한다. 이번에 못 쟀으면 이전 값을 유지
                if let d = det.distance {
                    let smoothed = track.distance.map { Float(smoothing) * d + Float(1 - smoothing) * $0 } ?? d
                    track.distance = smoothed
                    if let anchor = track.anchor {
                        if time - anchor.time >= speedInterval {
                            let speed = (anchor.distance - smoothed) / Float(time - anchor.time)
                            track.approach = track.approach.map { $0 * 0.5 + speed * 0.5 } ?? speed
                            track.anchor = (smoothed, time)
                        }
                    } else {
                        track.anchor = (smoothed, time)
                    }
                }
                track.hits += 1
                track.lastSeen = time
            } else {
                // 놓친 동안에는 움직이던 방향으로 옮기되 점점 느려지게 한다
                track.rect = predicted
                track.velocity = CGVector(dx: track.velocity.dx * 0.8, dy: track.velocity.dy * 0.8)
            }
            if !track.shown && track.hits >= minHits { track.shown = true }
            if time - track.lastSeen <= maxAge { next.append(track) }
        }
        // 짝이 없는 새 박스는 슬라이더 기준을 넘을 때만 새 트랙이 된다
        for (i, det) in detections.enumerated() where !used.contains(i) && det.confidence >= minConfidence {
            next.append(Track(id: UUID(), rect: det.rect, velocity: .zero, confidence: det.confidence,
                              labelScores: [det.label: det.confidence], distance: det.distance,
                              anchor: det.distance.map { ($0, time) }, approach: nil, hits: 1,
                              lastSeen: time, shown: false))
        }
        tracks = next
        return tracks.filter(\.shown)
            .map { Detection(id: $0.id, label: $0.label, confidence: $0.confidence, rect: $0.rect,
                             distance: $0.distance, seenFrames: $0.hits, approachSpeed: $0.approach) }
    }

    private func mix(_ a: CGRect, _ b: CGRect) -> CGRect {
        let s = smoothing
        return CGRect(x: a.minX + (b.minX - a.minX) * s, y: a.minY + (b.minY - a.minY) * s,
                      width: a.width + (b.width - a.width) * s, height: a.height + (b.height - a.height) * s)
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let i = inter.width * inter.height
        return i / (a.width * a.height + b.width * b.height - i)
    }
}
