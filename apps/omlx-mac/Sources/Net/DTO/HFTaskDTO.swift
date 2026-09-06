// PR 8 — HuggingFace download task + recommended-models surface.

import Foundation

struct HFTaskListResponse: Codable, Sendable {
    let tasks: [HFTaskDTO]
}

struct HFTaskDTO: Codable, Equatable, Sendable, Identifiable {
    let taskId: String
    let repoId: String
    let status: String
    let progress: Double
    let totalSize: Int64
    let downloadedSize: Int64
    let error: String
    let createdAt: Double
    let startedAt: Double
    let completedAt: Double
    let retryCount: Int
    /// Subfolder of the repo this task is fetching (e.g. `oQ4e`). `nil` — the
    /// only value older servers can produce, since the key is simply absent
    /// from their task JSON — means the task covers the whole repo. Declared
    /// Optional precisely so that absence decodes cleanly rather than
    /// throwing `keyNotFound`; the ModelScope pipeline reuses this type
    /// (see MSTaskDTO.swift) and never sets it.
    let variant: String?

    var id: String { taskId }

    /// Trimmed variant, or nil when the task is a whole-repo download.
    /// Collapses `nil` and `""` into one "no variant" case so view code
    /// doesn't have to test both.
    var variantOrNil: String? {
        guard let v = variant?.trimmingCharacters(in: .whitespacesAndNewlines),
              !v.isEmpty else { return nil }
        return v
    }

    enum Status: String {
        case pending, downloading, completed, failed, cancelled, paused
    }

    var statusEnum: Status? { Status(rawValue: status) }
    var isActive: Bool {
        statusEnum == .pending || statusEnum == .downloading
    }
}

struct StartHFDownloadRequest: Encodable, Sendable {
    let repoId: String
    let hfToken: String
    /// Optional single subfolder to fetch instead of the whole repo, e.g.
    /// `oQ4e`. Plain name, no slashes — it is *not* folded into `repoId`.
    ///
    /// The client's encoder is `.convertToSnakeCase`, and `variant` is one
    /// lowercase word, so it lands on the wire as the literal key `variant` —
    /// matching the server's pydantic field name without a CodingKeys remap
    /// (the same reason no other field in this file needs one).
    ///
    /// Left `nil` for a whole-repo download, and synthesized Encodable
    /// conformance omits nil Optionals entirely, so the request body is
    /// byte-identical to the pre-variant one in that case.
    let variant: String?
}

struct StartHFDownloadResponse: Decodable, Sendable {
    let success: Bool
    let task: HFTaskDTO?
}

// GET /admin/api/hf/recommended returns two parallel lists (mirrors
// hf_downloader.py:237-240). The dashboard JS paginates them separately;
// the Swift screen merges them into a single deduped list ordered
// trending-first.
struct HFRecommendedResponse: Codable, Sendable {
    let trending: [HFModelInfo]
    let popular: [HFModelInfo]
}

/// Response shape for GET /admin/api/hf/search?q=<query>. Backed by the same
/// HFModelInfo rows that /recommended returns, plus an optional `total`
/// count that the server populates when paginating.
struct HFSearchResponse: Codable, Sendable {
    let models: [HFModelInfo]
    let total: Int?
}

struct HFModelInfo: Codable, Equatable, Sendable, Identifiable {
    let repoId: String
    let name: String?
    let downloads: Int?
    let likes: Int?
    let trendingScore: Double?
    let size: Int64?
    let sizeFormatted: String?
    let params: Int64?
    let paramsFormatted: String?

    var id: String { repoId }
}
