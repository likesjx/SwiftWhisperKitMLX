import XCTest
@testable import SwiftWhisperKitMLX

final class WhisperTranscriberAvailabilityTests: XCTestCase {
    func testAliasResolves() {
        let t = WhisperTranscriber(options: .init())
        XCTAssertNotNil(t)
    }

    func testPrepareBehavior() async {
        let t = MLXWhisperTranscriber()
        #if canImport(MLX)
        XCTAssertNoThrow(try await t.prepare())
        #else
        do {
            try await t.prepare()
            XCTFail("Expected prepare() to throw when MLX unavailable")
        } catch {
            // Expected without MLX
        }
        #endif
    }
}
