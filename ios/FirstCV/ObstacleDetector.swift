import CoreGraphics
import simd

/// 걷는 길 앞에서 가장 가까운 장애물.
struct Obstacle: Equatable {
    let distance: Float  // 몸 앞으로 몇 m (땅과 나란한 거리)
    let point: CGPoint  // 그 장애물이 사진 어디에 있는지 (0~1, 원점 왼쪽 위). 이 점을 덮는 탐지 박스가 있으면 그 이름으로 부른다
    var label: String?
    let leftClear: Float  // 통로 바로 왼쪽·오른쪽 칸의 가장 가까운 장애물 거리. 없으면 .infinity
    let rightClear: Float
}

/// 걷는 길 앞에서 바닥이 꺼지는 곳 (내려가는 계단, 턱, 낭떠러지).
struct Drop: Equatable {
    let distance: Float  // 몸 앞으로 몇 m
    let point: CGPoint
    var label: String?  // 그 위치에 계단 박스가 있으면 "stairs"
}

/// 프레임 하나의 보행 판단 결과.
struct PathReport: Equatable {
    var obstacle: Obstacle?
    var drop: Drop?
    var walking = false  // 휴대폰 흔들림으로 본 걷는 중 여부 (다가오는 물체를 가릴 때 내 속도를 어림한다)
}

/// LiDAR 깊이 지도만으로 걷는 길을 막는 물체를 찾는다. 모델이 모르는 물건(기둥, 벽, 상자)도 잡힌다.
///
/// 깊이 지도의 점들을 3차원으로 되돌린 뒤 중력 방향으로 높이를 잰다.
/// - 바닥보다 15cm 이상 높고 머리 높이보다 낮으며
/// - 몸 앞 폭 70cm 통로 안에 있는 점만 장애물로 본다
/// - 잡음 한두 점에 반응하지 않도록 가까운 순서로 3번째 점의 거리를 쓴다
/// 같은 통로에서 바닥보다 15cm 이상 낮은 점이 보이면 낙차(Drop)로 본다. 반사되는 바닥의 잡음을 피하려고 5점 이상일 때만.
final class ObstacleDetector {
    let corridorHalfWidth: Float = 0.35
    let maxRange: Float = 3.0
    private let stepHeight: Float = 0.15  // 이보다 낮은 턱은 바닥으로 본다
    private let dropDepth: Float = 0.15  // 바닥보다 이만큼 낮으면 꺼진 곳
    private let headroom: Float = 0.4  // 휴대폰(가슴 높이)보다 이만큼 위까지는 머리에 부딪힌다고 본다
    private let cols = 24, rows = 32
    /// 휴대폰 기준 바닥 높이(m, 음수). 바닥이 보일 때마다 조금씩 고친다
    private(set) var floorHeight: Float = -1.2

    /// - gravity: 카메라 좌표(사진 오른쪽 x, 아래 y, 앞 z)로 옮긴 중력 방향
    /// - fovTan: 사진 긴 변(세로) 화각의 절반에 대한 tan
    /// - aspect: 사진 가로 / 세로
    /// - sample: 사진 좌표(0~1)의 깊이(m). 못 쟀으면 nil
    func update(gravity: SIMD3<Float>, fovTan: Float, aspect: Float,
                sample: (CGFloat, CGFloat) -> Float?) -> PathReport? {
        let g = simd_normalize(gravity)
        // 카메라가 보는 방향과 오른쪽 방향을 땅과 나란하게 눕힌다
        var forward = SIMD3<Float>(0, 0, 1) - g.z * g
        var right = SIMD3<Float>(1, 0, 0) - g.x * g
        guard simd_length(forward) > 0.3 else { return nil }  // 바닥을 거의 수직으로 내려다보면 앞을 알 수 없다
        forward = simd_normalize(forward)
        right = simd_normalize(right)

        var points: [(height: Float, ahead: Float, side: Float, at: CGPoint)] = []
        points.reserveCapacity(cols * rows)
        for i in 0..<cols {
            for j in 0..<rows {
                let px = (CGFloat(i) + 0.5) / CGFloat(cols), py = (CGFloat(j) + 0.5) / CGFloat(rows)
                guard let z = sample(px, py) else { continue }
                let p = SIMD3<Float>(Float(px - 0.5) * 2 * fovTan * aspect * z, Float(py - 0.5) * 2 * fovTan * z, z)
                points.append((-simd_dot(p, g), simd_dot(p, forward), simd_dot(p, right), CGPoint(x: px, y: py)))
            }
        }
        guard points.count > 50 else { return nil }

        // 바닥: 지금 바닥 높이 근처의 점 중 발에 가장 가까운 쪽의 높이.
        // 가장 낮은 점을 바닥으로 보면, 내려가는 계단의 아래층이 바닥이 되어 정작 낙차를 놓친다
        let w = corridorHalfWidth
        let band = max(floorHeight - 0.5, -1.9) ... min(floorHeight + 0.3, -0.6)
        let ground = points.filter { abs($0.side) < 3 * w && $0.ahead > 0 && band.contains($0.height) }
            .sorted { $0.ahead < $1.ahead }
        if ground.count >= 10 {
            let near = ground.prefix(max(10, ground.count / 4)).map(\.height).sorted()
            // 낮아지는 쪽은 한 번에 5mm(1초에 약 0.3m)까지만 따라간다. 발 앞 바닥은 화면에 안 보일 때가 많아서,
            // 내려가는 계단 앞에 서면 보이는 바닥이 전부 아래층이다. 바로 따라가면 낙차가 금방 사라진다.
            // 대신 계단을 다 내려온 뒤 몇 초 동안은 바닥을 높게 봐서 낙차 경고가 남을 수 있다
            floorHeight = max(floorHeight * 0.8 + near[near.count / 2] * 0.2, floorHeight - 0.005)
        }

        let blocking = points.filter {
            $0.height > floorHeight + stepHeight && $0.height < headroom && $0.ahead > 0.1 && $0.ahead < maxRange
        }
        func nearest(_ lane: ClosedRange<Float>) -> (distance: Float, at: CGPoint)? {
            let inLane = blocking.filter { lane.contains($0.side) }.sorted { $0.ahead < $1.ahead }
            return inLane.count >= 3 ? (inLane[2].ahead, inLane[2].at) : nil
        }
        var report = PathReport()
        if let hit = nearest(-w ... w) {
            report.obstacle = Obstacle(distance: hit.distance, point: hit.at,
                                       leftClear: nearest(-3 * w ... -w)?.distance ?? .infinity,
                                       rightClear: nearest(w ... 3 * w)?.distance ?? .infinity)
        }
        let below = points.filter {
            abs($0.side) <= w && $0.height < floorHeight - dropDepth && $0.ahead > 0.3 && $0.ahead < maxRange
        }.sorted { $0.ahead < $1.ahead }
        if below.count >= 5 {
            report.drop = Drop(distance: below[2].ahead, point: below[2].at)
        }
        return report
    }
}
