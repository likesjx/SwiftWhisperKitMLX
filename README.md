<h1 align="center">SwiftWhisperKitMLX</h1>
<p align="center"><a href="https://github.com/likesjx/SwiftWhisperKitMLX/actions/workflows/ci.yml"><img src="https://github.com/likesjx/SwiftWhisperKitMLX/actions/workflows/ci.yml/badge.svg" alt="CI"></a></p>
<p align="center">Experimental on-device Whisper (encoder + greedy decoder) implemented with MLX in Swift.</p>

> Status: Pre-0.1.0 – API & accuracy not stable. Tokenizer simplified. Timestamps & advanced decoding not yet implemented.

## Features (Current)
- Manifest-driven weight loader (Whisper encoder + decoder tensors)
- Simplified tokenizer with special tokens
- Encoder forward pass (transformer blocks)
- Greedy decoder (no timestamps, no beam search yet)

## Roadmap
- [ ] Full BPE merges + byte fallback
- [ ] Timestamp token handling & segmentation
- [ ] Beam search / temperature / top-p sampling
- [ ] Incremental streaming with KV cache reuse
- [ ] Quantization (int8 / mixed precision)
- [ ] Weight download & checksum verification
- [ ] Benchmarks & memory profiling

## Quick Start (Local Weights Already Installed)
```swift
import SwiftWhisperKitMLX

let transcriber = MLXWhisperTranscriber()
try await transcriber.prepare()
let result = try await transcriber.transcribe(samples: audioSamples, sampleRate: 44_100)
print("Transcribed:", result.text)
```

## Weight Layout
Expected directory (first found among Application Support & Documents paths):
```
~/Library/Application Support/MLXModels/<variant>/
	manifest.json
	vocab.json
	tokenizer.vocab.json (optional)
	tokenizer.merges.txt (optional)
	*.bin (tensor shards referenced in manifest)
```

You can symlink or copy model folders produced by your conversion pipeline. A helper script now exists to fetch weights.

## Downloading Weights
Use the helper script to fetch a hosted model variant (example variant `small`) where the layout is `<base>/<variant>/manifest.json` and related tensor shard files.

```bash
./Scripts/download-weights.sh small https://your-host.example/whisper
```

Outputs are placed in `~/Library/Application Support/<variant>` on macOS.

## Continuous Integration
GitHub Actions (macOS 14) builds & tests on pushes and pull requests. See badge above.

## Backwards Compatibility
`WhisperTranscriber` is deprecated; use `MLXWhisperTranscriber`. A typealias is provided to avoid breaking existing code.

## Limitations
- Tokenizer does not yet perform full Whisper BPE merge logic.
- No timestamp tokens or alignment; returns a single text string.
- Greedy decoding only – may produce suboptimal text.

## Contributing
See `CONTRIBUTING.md` for guidelines. PRs improving tokenizer fidelity, timestamps, or download manager are welcome.

## License
MIT – see `LICENSE`.

## Disclaimer
This is experimental and not production-quality. Validate outputs against a reference implementation before relying on accuracy.
