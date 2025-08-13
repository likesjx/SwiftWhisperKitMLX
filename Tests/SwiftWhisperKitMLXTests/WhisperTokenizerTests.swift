import XCTest
@testable import SwiftWhisperKitMLX

final class WhisperTokenizerTests: XCTestCase {
    func testSimpleEncodeDecode() throws {
        // Create temporary vocab & merges
        let vocabURL = FileManager.default.temporaryDirectory.appendingPathComponent("vocab.json")
        let mergesURL = FileManager.default.temporaryDirectory.appendingPathComponent("merges.txt")
        let vocab: [String:Int] = ["h":0,"e":1,"l":2,"o":3,"he":4,"llo":5]
        let data = try JSONEncoder().encode(vocab)
        try data.write(to: vocabURL)
        try "h e\ne l\nl l\nll o\n".write(to: mergesURL, atomically: true, encoding: .utf8)
        let tokenizer = try WhisperTokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
        let ids = tokenizer.encode("hello")
        let text = tokenizer.decode(ids)
        XCTAssertFalse(ids.isEmpty)
        XCTAssertEqual(text.first, "h")
    }
}
