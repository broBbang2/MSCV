/// 모델이 내는 영어 클래스 이름 → 화면에 보여줄 한국어 이름.
/// 학습 데이터·기록(csv)·맥 스크립트는 영어 이름을 그대로 쓰고, 화면 표시만 바꾼다.
/// 모르는 이름(클래스를 새로 추가했는데 여기 안 넣은 경우)은 영어 그대로 보여준다.
enum KoreanLabels {
    private static let names: [String: String] = [
        "monitor": "모니터",
        "laptop": "노트북",
        "keyboard": "키보드",
        "figure": "피규어",
        "toy car": "장난감 자동차",
        "tumbler": "텀블러",
        "bottle": "병",
        "speaker": "스피커",
        "headphones": "헤드폰",
        "clothes": "옷",
        "cap": "모자",
        "book": "책",
        "mouse": "마우스",
        "phone": "휴대폰",
        "cup": "컵",
        "shoes": "신발",
        "backpack": "백팩",
        "chair": "의자",
        "sofa": "소파",
        "person": "사람",
        "door": "문",
        "stairs": "계단",
        "bicycle": "자전거",
        "car": "차",
        "motorcycle": "오토바이",
        "pole": "기둥",
        "table": "테이블",
        "trash can": "쓰레기통",
        "bench": "벤치",
        "traffic cone": "라바콘",
    ]

    static func name(for label: String) -> String {
        names[label] ?? label
    }
}
