"use client";

import { use, useEffect, useState } from "react";
import { api, RCAReport } from "@/lib/api";
import Link from "next/link";
import { useResolvedInvestigationId } from "@/lib/useInvestigationId";
import { NOT_FOUND_MESSAGE, useInvestigationStatus } from "@/lib/useInvestigationStatus";
import { ConfidenceBar, EmptyState, PageHeader, Panel } from "@/components/ui";

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const investigationId = useResolvedInvestigationId(id);
  const [report, setReport] = useState<RCAReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lifecycle = useInvestigationStatus(investigationId);

  useEffect(() => {
    if (!investigationId) return;
    if (lifecycle.state === "error") setError(lifecycle.error);
    if (lifecycle.state !== "has_report") return; // no 404 probing before a report exists
    api
      .getReport(investigationId)
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [investigationId, lifecycle.state, lifecycle.error]);

  return (
    <div>
      <PageHeader
        eyebrow="RCA Report"
        title="Root Cause Analysis Report"
        description="Executive-ready summary generated from hybrid retrieval, graph reasoning, and LLM analysis."
        action={
          <button
            onClick={() => window.print()}
            className="rounded-md border border-line px-4 py-2 text-[13px] text-ink hover:bg-panel2"
          >
            Export / Print
          </button>
        }
      />

      <div className="mx-auto max-w-3xl space-y-6 px-8 py-8">
        {error && (
          <Panel className="border-critical/40 p-5 text-[13px] text-critical">{error}</Panel>
        )}
        {lifecycle.state === "not_found" && (
          <EmptyState
            title="Investigation not found"
            description={NOT_FOUND_MESSAGE}
            action={<Link href="/upload" className="text-[13px] text-signal underline">Upload logs</Link>}
          />
        )}
        {lifecycle.state === "no_report" && (
          <EmptyState
            title="No report yet"
            description="The RCA report is generated when the investigation runs."
            action={
              <Link href={`/investigate/${investigationId}`} className="text-[13px] text-signal underline">
                Open the workspace to run the investigation
              </Link>
            }
          />
        )}

        {report && (
          <>
            <Panel className="p-6">
              <p className="mb-2 text-[11px] font-medium text-faint">Executive Summary</p>
              <p className="text-[14px] leading-relaxed text-ink">{report.executive_summary}</p>
              <p className="mt-4 text-[11px] text-faint">
                Generated {new Date(report.generated_at).toLocaleString()}
              </p>
            </Panel>

            <Panel className="p-6">
              <p className="mb-2 text-[11px] font-medium text-faint">
                Root Cause
                {report.root_cause.root_cause_type ? ` · ${report.root_cause.root_cause_type}` : ""}
                {report.degraded ? " · deterministic analysis (Gemini unavailable)" : ""}
              </p>
              <p className="text-[16px] font-semibold text-ink">{report.root_cause.root_cause}</p>
              <div className="mt-4 max-w-xs">
                <ConfidenceBar value={report.confidence} />
              </div>
              {report.root_cause.confidence_explanation && (
                <p className="mt-2 text-[12px] leading-relaxed text-muted">
                  Confidence {report.root_cause.confidence_explanation}
                </p>
              )}
              <p className="mb-2 mt-5 text-[11px] font-medium text-faint">Cause chain</p>
              <div className="flex flex-wrap items-center gap-2">
                {report.root_cause.cause_chain.map((step, i) => (
                  <span key={i} className="flex items-center gap-2">
                    <span className="rounded border border-line bg-panel2 px-2 py-1 font-mono text-[12px] text-ink">
                      {step}
                    </span>
                    {i < report.root_cause.cause_chain.length - 1 && <span className="text-faint">→</span>}
                  </span>
                ))}
              </div>
            </Panel>

            <Panel className="p-6">
              <p className="mb-3 text-[11px] font-medium text-faint">
                Timeline ({report.timeline.events.length} events)
              </p>
              <div className="space-y-3">
                {report.timeline.events.slice(0, 8).map((e, i) => (
                  <div key={i} className="flex gap-3 text-[13px]">
                    <span className="w-16 shrink-0 font-mono text-[11px] text-signal">
                      {e.timestamp || `#${i + 1}`}
                    </span>
                    <span className="text-muted">{e.title}</span>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel className="p-6">
              <p className="mb-3 text-[11px] font-medium text-faint">Affected Systems</p>
              <div className="flex flex-wrap gap-2">
                {report.affected_systems.length === 0 && (
                  <span className="text-[13px] text-muted">Not conclusively identified.</span>
                )}
                {report.affected_systems.map((sys, i) => (
                  <span key={i} className="rounded-full bg-critical/10 px-3 py-1 text-[12px] text-critical">
                    {sys}
                  </span>
                ))}
              </div>
            </Panel>

            <Panel className="p-6">
              <p className="mb-3 text-[11px] font-medium text-faint">Evidence</p>
              <div className="space-y-3">
                {report.root_cause.evidence.slice(0, 5).map((ev, i) => (
                  <div key={i} className="rounded-md border border-line bg-panel2 p-3">
                    <p className="mb-1 text-[11px] text-faint">{ev.source_document}</p>
                    <p className="font-mono text-[12px] leading-relaxed text-muted">{ev.text}</p>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel className="p-6">
              <p className="mb-3 text-[11px] font-medium text-faint">Recommendations</p>
              <ul className="space-y-2">
                {report.recommendations.map((rec, i) => (
                  <li key={i} className="flex gap-2 text-[13px] text-ink">
                    <span className="text-trace">—</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}
