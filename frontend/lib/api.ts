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
}

export interface InvestigationSummary {
  id: string;
  title: string;
  status: string;
  created_at: string;
  document_count: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || `Request failed: ${res.status}`);
  }
  return res.json();
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
    request<ChatResponseDto>("/chat", {
      method: "POST",
      body: JSON.stringify({ investigation_id: investigationId, message, history }),
    }),

  listInvestigations: () => request<InvestigationSummary[]>("/investigations"),
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
