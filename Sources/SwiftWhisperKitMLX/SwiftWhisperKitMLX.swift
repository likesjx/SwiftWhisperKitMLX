// SwiftWhisperKitMLX umbrella file.
// The original placeholder types have been removed in favor of the real
// implementation in `WhisperTranscriber.swift`.
// Backward compatibility shims (if any early code imported these symbols)
// can be added here. For now we purposely leave it minimal.
//
// If external users previously used `WhisperTranscriber` (placeholder), they
// should migrate to `MLXWhisperTranscriber` and `WhisperLoadOptions` defined
// alongside the concrete implementation.

@_exported import Foundation
