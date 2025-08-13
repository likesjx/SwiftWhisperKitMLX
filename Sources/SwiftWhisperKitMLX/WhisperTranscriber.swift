import Foundation
#if canImport(MLX)
import MLX
import MLXNN
#endif

public struct WhisperTranscribeResult: Sendable { public let text:String; public let tokens:[Int] }
public struct WhisperLoadOptions: Sendable { public var variant: WhisperVariant; public init(variant: WhisperVariant = .auto){ self.variant = variant } }

public final class MLXWhisperTranscriber: @unchecked Sendable {
    private let options: WhisperLoadOptions
    private var weights: WhisperWeights?
    private var prepared = false
    private var tokenizer: WhisperTokenizer?
    public init(options: WhisperLoadOptions = .init()) { self.options = options }
    public func prepare() async throws {
        #if canImport(MLX)
        if prepared { return }
        let paths: [URL] = [FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!, FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!]
        self.weights = try WhisperLoader.load(variant: options.variant, preferredPaths: paths)
        if let modelDir = paths.compactMap({ $0.appendingPathComponent(options.variant.raw) }).first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            let vocabURL = modelDir.appendingPathComponent("tokenizer.vocab.json")
            let mergesURL = modelDir.appendingPathComponent("tokenizer.merges.txt")
            if FileManager.default.fileExists(atPath: vocabURL.path), FileManager.default.fileExists(atPath: mergesURL.path) {
                tokenizer = try? WhisperTokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
            }
        }
        prepared = true
        #else
        throw WhisperModelError.modelUnavailable
        #endif
    }
    public func transcribe(samples: [Float], sampleRate: Double) async throws -> WhisperTranscribeResult {
        #if canImport(MLX)
        guard prepared, let w = weights else { throw WhisperModelError.modelUnavailable }
        let mono = resampleTo16k(samples: samples, sampleRate: sampleRate, target: 16000)
        let mel = computeLogMel(samples: mono, nMels: w.config.nMels)
    // Encoder forward producing context representation
    let encoderHidden = runEncoder(mel: mel, weights: w)
    let promptTokens: [Int] = buildPromptTokens(tokenizer: tokenizer)
    // Run greedy decoder for limited steps
    let gen = runGreedyDecoder(encoderHidden: encoderHidden, prompt: promptTokens, weights: w, tokenizer: tokenizer, maxNew: 32)
    let tokens = promptTokens + gen
    let text: String
    if let tok = tokenizer { text = tok.decode(tokens) } else { text = tokens.compactMap { $0 < w.vocab.count ? w.vocab[$0] : nil }.joined(separator: " ") }
        return WhisperTranscribeResult(text: text, tokens: tokens)
        #else
        throw WhisperModelError.modelUnavailable
        #endif
    }
}

#if canImport(MLX)
private func runEncoder(mel: [[Float]], weights w: WhisperWeights) -> Tensor {
    guard !mel.isEmpty else { return Tensor(zeros: [1, w.config.dModel]) }
    let frames = min(mel.count, w.config.maxFrames)
    let melBins = w.config.nMels
    let flat = mel.prefix(frames).flatMap { $0 }
    var x = Tensor(shape: [frames, melBins], scalars: flat)
    x = (x • w.melProjW) + w.melProjB
    let pos = w.posEnc[0..<frames]
    x = x + pos
    for b in w.encBlocks { x = encoderBlock(x: x, b: b, cfg: w.config) }
    x = layerNorm(x: x, scale: w.encLNPostW, bias: w.encLNPostB) // (T,D)
    return x
}

// Simplified greedy decoder: ignores timestamps & task logic, stops on EOT or maxNew
private func runGreedyDecoder(encoderHidden: Tensor, prompt: [Int], weights w: WhisperWeights, tokenizer: WhisperTokenizer?, maxNew: Int) -> [Int] {
    var generated: [Int] = []
    let cfg = w.config
    var tokens = prompt
    var kvSelfK: [Tensor] = Array(repeating: Tensor(zeros:[0,0,0]), count: w.decBlocks.count)
    var kvSelfV: [Tensor] = Array(repeating: Tensor(zeros:[0,0,0]), count: w.decBlocks.count)
    let encKVCached: [(Tensor,Tensor)] = w.decBlocks.enumerated().map { _ in (encoderHidden, encoderHidden) } // cross-attn keys/values simplified
    let eot = tokenizer?.specials.eot
    for _ in 0..<maxNew {
        // Build token embedding sequence
        let seqLen = tokens.count
        var x = embed(tokens: tokens, w: w)
        if seqLen <= w.posDecEmbedding.shape[0] { x = x + w.posDecEmbedding[0..<seqLen] }
        // Decoder layers
        for (i, b) in w.decBlocks.enumerated() {
            (x, kvSelfK[i], kvSelfV[i]) = decoderBlock(x: x, b: b, cfg: cfg, pastK: kvSelfK[i], pastV: kvSelfV[i], enc: encoderHidden, crossKV: encKVCached[i])
        }
        x = layerNorm(x: x, scale: w.decLNPostW, bias: w.decLNPostB)
        // Last token logits
        let last = x[x.shape[0]-1]
        let proj = w.outputProj ?? w.tokenEmbedding.transposed(1,0)
        let logits = (last • proj)
        let nextId = logits.argmax().item()
        if let e = eot, nextId == e { break }
        generated.append(Int(nextId))
        tokens.append(Int(nextId))
    }
    return generated
}

private func encoderBlock(x: Tensor, b: WhisperWeights.EncBlock, cfg: WhisperConfig) -> Tensor {
    var h = layerNorm(x: x, scale: b.ln1W, bias: b.ln1B)
    h = h + selfAttention(x: h, b: b, cfg: cfg)
    var f = layerNorm(x: h, scale: b.ln2W, bias: b.ln2B)
    f = gelu((f • b.f1) + b.f1B)
    f = (f • b.f2) + b.f2B
    return h + f
}

private func selfAttention(x: Tensor, b: WhisperWeights.EncBlock, cfg: WhisperConfig) -> Tensor {
    let d = cfg.dModel; let h = cfg.nHeads; let headDim = d / h
    let q = ((x • b.q) + b.qB).reshaped([-1,h,headDim])
    let k = ((x • b.k) + b.kB).reshaped([-1,h,headDim])
    let v = ((x • b.v) + b.vB).reshaped([-1,h,headDim])
    var attn = (q * (1.0 / sqrt(Float(headDim)))) • k.transposed(0,2,1)
    attn = attn.softmax(axis: 2)
    var out = attn • v
    out = out.reshaped([-1,d])
    return (out • b.o) + b.oB
}

// Decoder components (minimal, reuse some helper ops)
private func decoderBlock(x: Tensor, b: WhisperWeights.DecBlock, cfg: WhisperConfig, pastK: Tensor, pastV: Tensor, enc: Tensor, crossKV: (Tensor,Tensor)) -> (Tensor, Tensor, Tensor) {
    var h = layerNorm(x: x, scale: b.ln1W, bias: b.ln1B)
    // Self-attn with causal mask
    let (sa, kNew, vNew) = causalSelfAttn(x: h, b: b, cfg: cfg, pastK: pastK, pastV: pastV)
    h = x + sa
    var c = layerNorm(x: h, scale: b.ln2W, bias: b.ln2B)
    c = h + crossAttn(x: c, b: b, cfg: cfg, enc: enc, cross: crossKV)
    var f = layerNorm(x: c, scale: b.ln3W, bias: b.ln3B)
    f = gelu((f • b.f1) + b.f1B)
    f = (f • b.f2) + b.f2B
    return (c + f, kNew, vNew)
}

private func causalSelfAttn(x: Tensor, b: WhisperWeights.DecBlock, cfg: WhisperConfig, pastK: Tensor, pastV: Tensor) -> (Tensor, Tensor, Tensor) {
    let d = cfg.dModel; let h = cfg.nHeads; let headDim = d / h
    let q = ((x • b.selfQ) + b.selfQB).reshaped([-1,h,headDim])
    let kNew = ((x • b.selfK) + b.selfKB).reshaped([-1,h,headDim])
    let vNew = ((x • b.selfV) + b.selfVB).reshaped([-1,h,headDim])
    let kCat = concat([pastK, kNew], axis: 0)
    let vCat = concat([pastV, vNew], axis: 0)
    var attn = (q * (1.0 / sqrt(Float(headDim)))) • kCat.transposed(0,2,1)
    // causal mask: large negative for future positions
    let seqQ = q.shape[0]; let seqK = kCat.shape[0]
    // Build mask lazily (O(n^2) small for our greedy loop)
    var maskVals = [Float](repeating: 0, count: seqQ*seqK)
    for i in 0..<seqQ { for j in 0..<seqK where j > (pastK.shape[0] + i) { maskVals[i*seqK + j] = -1e4 } }
    let mask = Tensor(shape: [seqQ, seqK], scalars: maskVals).reshaped([seqQ,1,seqK])
    attn = attn + mask
    attn = attn.softmax(axis: 2)
    var out = attn • vCat
    out = out.reshaped([-1,d])
    out = (out • b.selfO) + b.selfOB
    return (out, kCat, vCat)
}

private func crossAttn(x: Tensor, b: WhisperWeights.DecBlock, cfg: WhisperConfig, enc: Tensor, cross: (Tensor,Tensor)) -> Tensor {
    let d = cfg.dModel; let h = cfg.nHeads; let headDim = d / h
    let q = ((x • b.crossQ) + b.crossQB).reshaped([-1,h,headDim])
    let k = ((enc • b.crossK) + b.crossKB).reshaped([-1,h,headDim])
    let v = ((enc • b.crossV) + b.crossVB).reshaped([-1,h,headDim])
    var attn = (q * (1.0 / sqrt(Float(headDim)))) • k.transposed(0,2,1)
    attn = attn.softmax(axis: 2)
    var out = attn • v
    out = out.reshaped([-1,d])
    return (out • b.crossO) + b.crossOB
}

private func embed(tokens: [Int], w: WhisperWeights) -> Tensor {
    let d = w.config.dModel
    var scalars: [Float] = []
    scalars.reserveCapacity(tokens.count * d)
    for id in tokens { let start = id * d; let slice = w.tokenEmbedding[start..<(start+d)]; scalars.append(contentsOf: slice.asArray()) }
    return Tensor(shape: [tokens.count, d], scalars: scalars)
}

private func layerNorm(x: Tensor, scale: Tensor, bias: Tensor, eps: Float = 1e-5) -> Tensor { let mean = x.mean(axis: -1, keepDims: true); let varT = ((x-mean)*(x-mean)).mean(axis:-1, keepDims:true); let norm = (x-mean)/(varT+eps).sqrt(); return norm * scale + bias }
private func gelu(_ t: Tensor) -> Tensor { 0.5 * t * (1 + ((t / sqrt(2.0)).erf())) }
private func softmax(_ t: Tensor) -> Tensor { let m = t.max(); let e = (t - m).exp(); return e / e.sum() }

private func resampleTo16k(samples: [Float], sampleRate: Double, target: Double) -> [Float] {
    if sampleRate == target { return samples }
    let ratio = target / sampleRate
    let outCount = Int(Double(samples.count) * ratio)
    var output = [Float](repeating: 0, count: outCount)
    for i in 0..<outCount { let pos = Double(i)/ratio; let i0 = Int(pos); let frac = Float(pos - Double(i0)); if i0+1 < samples.count { output[i] = samples[i0]*(1-frac) + samples[i0+1]*frac } else { output[i] = samples.last ?? 0 } }
    return output
}

private func computeLogMel(samples: [Float], nMels: Int) -> [[Float]] {
    let fftSize = 400; let hop = 160; guard samples.count >= fftSize else { return [] }
    let frameCount = (samples.count - fftSize)/hop + 1
    var hann = [Float](repeating:0,count:fftSize)
    vDSP_hann_window(&hann, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
    let bins = fftSize/2
    var melSpec:[[Float]] = []; melSpec.reserveCapacity(frameCount)
    let melFilters = buildSimpleMelFilters(mels: nMels, bins: bins)
    var real = [Float](repeating:0,count:bins)
    var imag = [Float](repeating:0,count:bins)
    var split = DSPSplitComplex(realp: &real, imagp: &imag)
    let setup = vDSP_create_fftsetup(vDSP_Length(log2(Float(fftSize))), FFTRadix(FFT_RADIX2))
    defer { if let s=setup { vDSP_destroy_fftsetup(s) } }
    for f in 0..<frameCount {
        let off = f*hop
        var frame = Array(samples[off..<off+fftSize])
        vDSP_vmul(frame,1,hann,1,&frame,1,vDSP_Length(fftSize))
        frame.withUnsafeMutableBufferPointer { buf in buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftSize/2) { ptr in vDSP_ctoz(ptr,2,&split,1,vDSP_Length(fftSize/2)) }; if let s=setup { vDSP_fft_zrip(s,&split,1,vDSP_Length(log2(Float(fftSize))),FFTDirection(FFT_FORWARD)) } }
        var mags = [Float](repeating:0,count:bins); vDSP_zvmags(&split,1,&mags,1,vDSP_Length(bins))
        var mel = [Float](repeating:0,count:nMels)
        for m in 0..<nMels { let filt = melFilters[m]; let c = min(filt.count, mags.count); var dot:Float = 0; vDSP_dotpr(filt,1,mags,1,&dot,vDSP_Length(c)); mel[m] = log10(dot + 1e-6) }
        melSpec.append(mel)
    }
    return melSpec
}

private func buildSimpleMelFilters(mels:Int, bins:Int) -> [[Float]] { var filters:[[Float]] = []; filters.reserveCapacity(mels); for m in 0..<mels { var f=[Float](repeating:0,count:bins); let start = m*(bins/mels); let end = (m+1)*(bins/mels); if end>start { for i in start..<end { f[i] = Float(i-start)/Float(max(1,end-start)) } }; filters.append(f) }; return filters }
#endif

#if canImport(MLX)
private func buildPromptTokens(tokenizer: WhisperTokenizer?) -> [Int] {
    guard let t = tokenizer else { return [] }
    var ids: [Int] = []
    if let sot = t.specials.sot { ids.append(sot) }
    if let transcribe = t.specials.transcribe { ids.append(transcribe) }
    return ids
}
#endif
