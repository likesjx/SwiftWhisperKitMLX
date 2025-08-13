# Contributing

Thanks for your interest in contributing to SwiftWhisperKitMLX!

## Development Setup
1. Clone the repository.
2. Open `Package.swift` in Xcode 15+ (or 16 beta) or use SwiftPM CLI.
3. Place model weights under `~/Library/Application Support/MLXModels/<variant>/` or provide a custom search path.

## Testing
Run:
```
swift test --parallel
```

## Pull Requests
- Keep PRs focused and small.
- Add tests for new functionality.
- Update CHANGELOG.md under an "Unreleased" section.

## Coding Style
- Follow Swift API Design Guidelines.
- Prefer explicit access control (`public` only where needed).

## Roadmap Items Welcomed
- Timestamp decoding
- Beam search / sampling strategies
- Quantization support
- Weight download & caching manager
