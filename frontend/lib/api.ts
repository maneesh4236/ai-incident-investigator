/**
 * Thin fetch wrapper around the FastAPI backend.
 * All investigation state (current investigation id) is kept in
 * localStorage so the workspace can be reopened across page reloads.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

export interface UploadedDocumentDto {
  id: string;
  filename: string;
  doc_type: string;
  investigation_id: string;
  size_bytes: number;
}

export interface UploadResponse {
  investigation_id: string;
  documents: UploadedDocumentDto[];
  total_chunks: number;
  status: string;
}

export interface Evidence {
  text: string;
  source_document: string;
  chunk_id: string;
  relevance: number;
}

export interface RootCauseResult {
  root_cause: string;
  cause_chain: string[];
  confidence_score: number;
  evidence: Evidence[];
  affected_systems: string[];
  root_cause_type?: "OBSERVED" | "INFERRED" | "LIKELY" | "CONFIRMED" | "UNKNOWN";
  confidence_label?: string | null;
  confidence_explanation?: string | null;
  source?: "gemini" | "deterministic";
}

export interface TimelineEvent {
  timestamp: string | null;
  order: number;
  title: string;
  description: string;
  severity: "info" | "warning" | "critical";
  source_chunk_ids: string[];
}

export interface Timeline {
  investigation_id: string;
  events: TimelineEvent[];
}

export interface RCAReport {
  investigation_id: string;
  generated_at: string;
  executive_summary: string;
  root_cause: RootCauseResult;
  timeline: Timeline;
  affected_systems: string[];
  degraded?: boolean;
  degradation_reason?: string | null;
  recommendations: string[];
  confidence: number;
}

export interface GraphEntity {
  id: string;
  name: string;
  type: string;
  investigation_id: string;
  source_chunk_ids: string[];
}

export interface GraphRelationship {
  id: string;
  source: string;
  target: string;
  type: string;
  investigation_id: string;
  confidence: number;
}

export interface KnowledgeGraphResponse {
  investigation_id: string;
  nodes: GraphEntity[];
  edges: GraphRelationship[];
}

export interface ChatMessageDto {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponseDto {
  answer: string;
  supporting_evidence: Evidence[];
  referenced_entities: string[];
  evidence_ids?: string[];
  reasoning_mode?: "gemini" | "deterministic_fallback";
  degraded?: boolean;
  degradation_reason?: string | null;
}

// The backend answers chat within its own Gemini deadline (CHAT_GEMINI_DEADLINE_SECONDS,
// default 35 s) and falls back deterministically; this client-side limit only guards
// against a backend that never responds, so the UI can never stay on "Investigating…".
const CHAT_TIMEOUT_MS = 90_000;

export interface InvestigationSummary {
  id: string;
  title: string;
  status: string;
  created_at: string;
  document_count: number;
}

/** Error carrying the HTTP status, so callers can treat expected states (e.g. 404) intentionally. */
export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiError";
  }
}

export function isNotFound(e: unknown): boolean {
  return e instanceof ApiError && e.status === 404;
}

export interface InvestigationStatusDto {
  id: string;
  title: string;
  status: string;
  created_at: string;
  document_count: number;
  has_report: boolean;
  degraded: boolean | null;
  error: string | null;
}

export interface MetaDto {
  app: string;
  llm: { provider: string; model: string; display_name: string; configured: boolean };
  vector_store: { backend: string; mode: string };
  graph_store: { backend: string; connected: boolean; unavailable_reason: string | null };
}

async function request<T>(path: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
  const controller = timeoutMs ? new AbortController() : undefined;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : undefined;
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller?.signal ?? init?.signal,
      headers: {
        ...(init?.body && !(init.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...init?.headers,
      },
    });
  } catch (e: any) {
    if (e?.name === "AbortError") {
      throw new Error("The server took too long to respond. Please try again.");
    }
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(describeDetail(body?.detail) || `Request failed: ${res.status}`, res.status);
  }
  return res.json();
}

function describeDetail(detail: unknown): string {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    return String(d.message ?? d.error ?? d.reason ?? "Request failed");
  }
  return String(detail);
}

export const api = {
  uploadDocuments: (files: File[], investigationId?: string, title?: string) => {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    if (investigationId) formData.append("investigation_id", investigationId);
    if (title) formData.append("title", title);
    return request<UploadResponse>("/upload", { method: "POST", body: formData });
  },

  investigate: (investigationId: string, question?: string) =>
    request<RCAReport>("/investigate", {
      method: "POST",
      body: JSON.stringify({ investigation_id: investigationId, question }),
    }),

  getGraph: (investigationId: string) =>
    request<KnowledgeGraphResponse>(`/graph/${investigationId}`),

  getTimeline: (investigationId: string) =>
    request<Timeline>(`/timeline/${investigationId}`),

  getReport: (investigationId: string) =>
    request<RCAReport>(`/report/${investigationId}`),

  chat: (investigationId: string, message: string, history: ChatMessageDto[]) =>
    request<ChatResponseDto>(
      "/chat",
      {
        method: "POST",
        body: JSON.stringify({ investigation_id: investigationId, message, history }),
      },
      CHAT_TIMEOUT_MS,
    ),

  listInvestigations: () => request<InvestigationSummary[]>("/investigations"),

  getInvestigation: (investigationId: string) =>
    request<InvestigationStatusDto>(`/investigations/${investigationId}`),

  getMeta: () => request<MetaDto>("/meta"),
};

const CURRENT_INVESTIGATION_KEY = "aether:current_investigation_id";

export function getCurrentInvestigationId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(CURRENT_INVESTIGATION_KEY);
}

export function setCurrentInvestigationId(id: string) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(CURRENT_INVESTIGATION_KEY, id);
}

/** Forget the stored investigation id (e.g. after the backend restarted and no longer knows it). */
export function clearCurrentInvestigationId(id?: string) {
  if (typeof window === "undefined") return;
  if (!id || window.localStorage.getItem(CURRENT_INVESTIGATION_KEY) === id) {
    window.localStorage.removeItem(CURRENT_INVESTIGATION_KEY);
  }
}
