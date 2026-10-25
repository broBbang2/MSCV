import CoreGraphics
import Foundation

/// 프레임마다 따로 나오는 탐지 결과를 이어 붙여 화면에서 박스가 깜빡이거나 튀지 않게 한다.
///
/// - 같은 이름이고 많이 겹치는 박스는 같은 물건으로 보고, 위치를 이전 박스와 섞어 부드럽게 옮긴다
/// - 처음 보인 물건은 2프레임 연속 잡혀야 보여준다 (한 프레임짜리 오탐을 거른다)
/// - 잡히던 물건을 놓쳐도 6프레임(약 0.2초)까지는 마지막 위치에 남겨둔다
final class Tracker {
    private struct Track {
        let id: UUID
        let label: String
        var rect: CGRect
        var confidence: Float
        var distance: Float?
        var hits: Int
        var misses: Int
    }

    private var tracks: [Track] = []
    private let minHits = 2
    private let maxMisses = 6
    private let minIoU: CGFloat = 0.3
    private let smoothing: CGFloat = 0.5  // 새 박스를 얼마나 반영할지 (1이면 그대로, 작을수록 부드럽고 느리다)

    func update(_ detections: [Detection]) -> [Detection] {
        var used = Set<Int>()
        var next: [Track] = []

        // 점수 높은 트랙부터 같은 이름의 가장 많이 겹치는 새 박스와 짝짓는다
        for var track in tracks.sorted(by: { $0.confidence > $1.confidence }) {
            var best: Int?
            var bestIoU = minIoU
            for (i, det) in detections.enumerated() where !used.contains(i) && det.label == track.label {
                let v = iou(track.rect, det.rect)
                if v > bestIoU { bestIoU = v; best = i }
            }
            if let best {
                used.insert(best)
                track.rect = mix(track.rect, detections[best].rect)
                track.confidence = Float(smoothing) * detections[best].confidence + Float(1 - smoothing) * track.confidence
                // 거리도 박스처럼 섞어서 숫자가 덜 흔들리게 한다. 이번에 못 쟀으면 이전 값을 유지
                if let d = detections[best].distance {
                    track.distance = track.distance.map { Float(smoothing) * d + Float(1 - smoothing) * $0 } ?? d
                }
                track.hits += 1
                track.misses = 0
            } else {
                track.misses += 1
            }
            if track.misses <= maxMisses { next.append(track) }
        }
        for (i, det) in detections.enumerated() where !used.contains(i) {
            next.append(Track(id: UUID(), label: det.label, rect: det.rect, confidence: det.confidence,
                              distance: det.distance, hits: 1, misses: 0))
        }
        tracks = next
        return tracks.filter { $0.hits >= minHits }
            .map { Detection(id: $0.id, label: $0.label, confidence: $0.confidence, rect: $0.rect, distance: $0.distance) }
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
