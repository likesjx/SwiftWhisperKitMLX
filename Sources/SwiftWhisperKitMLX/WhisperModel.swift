import Foundation
#if canImport(MLX)
import MLX
import MLXNN
import MLXRandom
#endif

public enum WhisperModelError: Error, Sendable { case modelUnavailable, manifestMissing, tensorMissing(String), shapeMismatch(String) }

public struct WhisperConfig: Sendable { public let nMels:Int; public let dModel:Int; public let nHeads:Int; public let nEncoderLayers:Int; public let nDecoderLayers:Int; public let vocabSize:Int; public let maxFrames:Int; public let maxTextTokens:Int }
public struct WhisperVariant: Sendable, Hashable { public let raw:String; public init(_ raw:String){ self.raw=raw }; public static let auto = WhisperVariant("auto") }

#if canImport(MLX)
public final class WhisperWeights: @unchecked Sendable {
    public let config: WhisperConfig; public let vocab:[String]
    public let melProjW:Tensor; public let melProjB:Tensor; public let posEnc:Tensor
    public struct EncBlock { let q:Tensor; let k:Tensor; let v:Tensor; let o:Tensor; let qB:Tensor; let kB:Tensor; let vB:Tensor; let oB:Tensor; let ln1W:Tensor; let ln1B:Tensor; let ln2W:Tensor; let ln2B:Tensor; let f1:Tensor; let f1B:Tensor; let f2:Tensor; let f2B:Tensor }
    public let encBlocks:[EncBlock]; public let encLNPostW:Tensor; public let encLNPostB:Tensor
    public struct DecBlock { let selfQ:Tensor; let selfK:Tensor; let selfV:Tensor; let selfO:Tensor; let selfQB:Tensor; let selfKB:Tensor; let selfVB:Tensor; let selfOB:Tensor; let crossQ:Tensor; let crossK:Tensor; let crossV:Tensor; let crossO:Tensor; let crossQB:Tensor; let crossKB:Tensor; let crossVB:Tensor; let crossOB:Tensor; let ln1W:Tensor; let ln1B:Tensor; let ln2W:Tensor; let ln2B:Tensor; let ln3W:Tensor; let ln3B:Tensor; let f1:Tensor; let f1B:Tensor; let f2:Tensor; let f2B:Tensor }
    public let tokenEmbedding:Tensor; public let posDecEmbedding:Tensor; public let decBlocks:[DecBlock]; public let decLNPostW:Tensor; public let decLNPostB:Tensor; public let outputProj:Tensor?
    public init(config:WhisperConfig, vocab:[String], melProjW:Tensor, melProjB:Tensor, posEnc:Tensor, encBlocks:[EncBlock], encLNPostW:Tensor, encLNPostB:Tensor, tokenEmbedding:Tensor, posDecEmbedding:Tensor, decBlocks:[DecBlock], decLNPostW:Tensor, decLNPostB:Tensor, outputProj:Tensor?) { self.config=config; self.vocab=vocab; self.melProjW=melProjW; self.melProjB=melProjB; self.posEnc=posEnc; self.encBlocks=encBlocks; self.encLNPostW=encLNPostW; self.encLNPostB=encLNPostB; self.tokenEmbedding=tokenEmbedding; self.posDecEmbedding=posDecEmbedding; self.decBlocks=decBlocks; self.decLNPostW=decLNPostW; self.decLNPostB=decLNPostB; self.outputProj=outputProj }
}

public enum WhisperLoader {
    struct Manifest: Decodable { struct Cfg: Decodable { let n_mels:Int; let d_model:Int; let n_heads:Int; let n_encoder_layers:Int; let n_decoder_layers:Int; let vocab_size:Int; let max_frames:Int; let max_text_tokens:Int? }; struct TensorE: Decodable { let shape:[Int]; let file:String }; let config:Cfg; let tensors:[String:TensorE] }
    public static func load(variant:WhisperVariant, preferredPaths:[URL]) throws -> WhisperWeights {
        guard let dir = locateVariantDir(variant: variant, candidates: preferredPaths) else { throw WhisperModelError.modelUnavailable }
        let manifestURL = dir.appendingPathComponent("manifest.json"); guard FileManager.default.fileExists(atPath: manifestURL.path) else { throw WhisperModelError.manifestMissing }
        let data = try Data(contentsOf: manifestURL); let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        let cfg = WhisperConfig(nMels: manifest.config.n_mels, dModel: manifest.config.d_model, nHeads: manifest.config.n_heads, nEncoderLayers: manifest.config.n_encoder_layers, nDecoderLayers: manifest.config.n_decoder_layers, vocabSize: manifest.config.vocab_size, maxFrames: manifest.config.max_frames, maxTextTokens: manifest.config.max_text_tokens ?? 448)
        let vocab = try loadVocab(dir: dir); func tensor(_ key:String) throws -> Tensor { guard let t = manifest.tensors[key] else { throw WhisperModelError.tensorMissing(key) }; return try loadTensor(file: dir.appendingPathComponent(t.file), shape: t.shape) }
        let melProjW = try tensor("encoder.mel_projection.weight"); let melProjB = try tensor("encoder.mel_projection.bias"); let posEnc = try tensor("encoder.positional_embedding")
        var encBlocks:[WhisperWeights.EncBlock] = []; for i in 0..<cfg.nEncoderLayers { let p="encoder.blocks.\(i)."; encBlocks.append(.init(q: try tensor(p+"attn.query.weight"), k: try tensor(p+"attn.key.weight"), v: try tensor(p+"attn.value.weight"), o: try tensor(p+"attn.out.weight"), qB: try tensor(p+"attn.query.bias"), kB: try tensor(p+"attn.key.bias"), vB: try tensor(p+"attn.value.bias"), oB: try tensor(p+"attn.out.bias"), ln1W: try tensor(p+"ln1.weight"), ln1B: try tensor(p+"ln1.bias"), ln2W: try tensor(p+"ln2.weight"), ln2B: try tensor(p+"ln2.bias"), f1: try tensor(p+"mlp.fc1.weight"), f1B: try tensor(p+"mlp.fc1.bias"), f2: try tensor(p+"mlp.fc2.weight"), f2B: try tensor(p+"mlp.fc2.bias")) ) }
        let encLNPostW = try tensor("encoder.ln_post.weight"); let encLNPostB = try tensor("encoder.ln_post.bias")
        let tokenEmbedding = try tensor("decoder.token_embedding.weight"); let posDecEmbedding = try tensor("decoder.positional_embedding")
        var decBlocks:[WhisperWeights.DecBlock] = []; for i in 0..<cfg.nDecoderLayers { let p="decoder.blocks.\(i)."; decBlocks.append(.init(selfQ: try tensor(p+"self_attn.query.weight"), selfK: try tensor(p+"self_attn.key.weight"), selfV: try tensor(p+"self_attn.value.weight"), selfO: try tensor(p+"self_attn.out.weight"), selfQB: try tensor(p+"self_attn.query.bias"), selfKB: try tensor(p+"self_attn.key.bias"), selfVB: try tensor(p+"self_attn.value.bias"), selfOB: try tensor(p+"self_attn.out.bias"), crossQ: try tensor(p+"cross_attn.query.weight"), crossK: try tensor(p+"cross_attn.key.weight"), crossV: try tensor(p+"cross_attn.value.weight"), crossO: try tensor(p+"cross_attn.out.weight"), crossQB: try tensor(p+"cross_attn.query.bias"), crossKB: try tensor(p+"cross_attn.key.bias"), crossVB: try tensor(p+"cross_attn.value.bias"), crossOB: try tensor(p+"cross_attn.out.bias"), ln1W: try tensor(p+"ln1.weight"), ln1B: try tensor(p+"ln1.bias"), ln2W: try tensor(p+"ln2.weight"), ln2B: try tensor(p+"ln2.bias"), ln3W: try tensor(p+"ln3.weight"), ln3B: try tensor(p+"ln3.bias"), f1: try tensor(p+"mlp.fc1.weight"), f1B: try tensor(p+"mlp.fc1.bias"), f2: try tensor(p+"mlp.fc2.weight"), f2B: try tensor(p+"mlp.fc2.bias")) ) }
        let decLNPostW = try tensor("decoder.ln_post.weight"); let decLNPostB = try tensor("decoder.ln_post.bias")
        let outputProj:Tensor? = try? tensor("decoder.output_projection.weight")
        return WhisperWeights(config: cfg, vocab: vocab, melProjW: melProjW, melProjB: melProjB, posEnc: posEnc, encBlocks: encBlocks, encLNPostW: encLNPostW, encLNPostB: encLNPostB, tokenEmbedding: tokenEmbedding, posDecEmbedding: posDecEmbedding, decBlocks: decBlocks, decLNPostW: decLNPostW, decLNPostB: decLNPostB, outputProj: outputProj)
    }
    private static func locateVariantDir(variant:WhisperVariant, candidates:[URL]) -> URL? { for c in candidates { let d = c.appendingPathComponent(variant.raw); if FileManager.default.fileExists(atPath: d.path) { return d } } ; return nil }
    private static func loadVocab(dir:URL) throws -> [String] { let v = dir.appendingPathComponent("vocab.json"); let data = try Data(contentsOf: v); return try JSONDecoder().decode([String].self, from: data) }
    private static func loadTensor(file:URL, shape:[Int]) throws -> Tensor { let data = try Data(contentsOf: file); let count = data.count / MemoryLayout<Float>.size; let expected = shape.reduce(1,*); guard count == expected else { throw WhisperModelError.shapeMismatch(file.lastPathComponent) }; let arr:[Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }; return Tensor(shape: shape, scalars: arr) }
}
#endif

#if !canImport(MLX)
// Fallback stubs so the package can build when MLX dependencies are not present.
// These provide only the minimal surface required by other files (e.g. optional properties).
public final class WhisperWeights: @unchecked Sendable {}
public enum WhisperLoader {}
#endif
