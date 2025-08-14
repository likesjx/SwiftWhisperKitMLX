import XCTest
@testable import SwiftWhisperKitMLX

final class WhisperTranscriberStubTests: XCTestCase {
    func testTypealiasAvailability() throws {
        // Ensure deprecated typealiases resolve
        _ = WhisperLoadOptions()
        let _ = WhisperTranscriber(options: .init())
    }
}
