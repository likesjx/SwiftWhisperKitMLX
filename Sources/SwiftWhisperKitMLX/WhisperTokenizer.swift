import Foundation

public struct WhisperToken: Hashable, Sendable { public let id: Int }

public enum WhisperTokenizerError: Error { case vocabMissing, mergesMissing }

public struct WhisperSpecialTokens: Sendable { public let sot:Int?; public let eot:Int?; public let translate:Int?; public let transcribe:Int? }

public final class WhisperTokenizer: @unchecked Sendable {
    private let vocab: [String:Int]
    private let merges: [Pair: Int]
    private let reverseVocab: [Int:String]
    private struct Pair: Hashable { let a: String; let b: String }
    public let specials: WhisperSpecialTokens
    
    public init(vocabURL: URL, mergesURL: URL) throws {
        let data = try Data(contentsOf: vocabURL)
        let decoded = try JSONDecoder().decode([String:Int].self, from: data)
        self.vocab = decoded
        self.reverseVocab = Dictionary(uniqueKeysWithValues: decoded.map { ($0.value, $0.key) })
        let mergesText = try String(contentsOf: mergesURL)
        var merges: [Pair:Int] = [:]
        var rank = 0
        for line in mergesText.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            merges[Pair(a: String(parts[0]), b: String(parts[1]))] = rank
            rank += 1
        }
        self.merges = merges
        self.specials = WhisperSpecialTokens(
            sot: decoded["<|startoftranscript|>"],
            eot: decoded["<|endoftext|>"],
            translate: decoded["<|translate|>"],
            transcribe: decoded["<|transcribe|>"]
        )
    }
    
    public func encode(_ text: String) -> [Int] {
        // Simplified byte-level + greedy merges; not production fidelity
        var tokens = text.map { String($0) }
        while true {
            var bestPair: (Pair, Int)?
            for i in 0..<max(0,tokens.count-1) {
                let p = Pair(a: tokens[i], b: tokens[i+1])
                if let r = merges[p], (bestPair == nil || r < bestPair!.1) { bestPair = (p, r) }
            }
            guard let pair = bestPair else { break }
            if let idx = tokens.firstIndex(where: { $0 == pair.0.a }) {
                if idx < tokens.count - 1 && tokens[idx+1] == pair.0.b { tokens.replaceSubrange(idx...idx+1, with: [pair.0.a+pair.0.b]) }
            }
        }
        return tokens.compactMap { vocab[$0] }
    }
    
    public func decode(_ ids: [Int]) -> String { ids.compactMap { reverseVocab[$0] }.joined() }
    public func tokenString(_ id:Int) -> String? { reverseVocab[id] }
}
