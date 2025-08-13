// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftWhisperKitMLX",
    platforms: [ .iOS(.v17), .macOS(.v14) ],
    products: [ .library(name: "SwiftWhisperKitMLX", targets: ["SwiftWhisperKitMLX"]) ],
    dependencies: [
        // MLX (uncomment when upstream public URL is stable)
        // .package(url: "https://github.com/ml-explore/mlx-swift.git", branch: "main")
    ],
    targets: [
        .target(
            name: "SwiftWhisperKitMLX",
            dependencies: [/*"MLX", "MLXNN"*/],
            path: "Sources",
            resources: [ .process("Resources") ]
        ),
        .testTarget(
            name: "SwiftWhisperKitMLXTests",
            dependencies: ["SwiftWhisperKitMLX"],
            path: "Tests"
        )
    ]
)
