"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { api, RCAReport, setCurrentInvestigationId } from "@/lib/api";
import { useResolvedInvestigationId } from "@/lib/useInvestigationId";
import { ConfidenceBar, PageHeader, Panel, PrimaryButton } from "@/components/ui";

export default function WorkspacePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const investigationId = useResolvedInvestigationId(id);
  const [question, setQuestion] = useState("");
  const [report, setReport] = useState<RCAReport | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!investigationId) return;
    setCurrentInvestigationId(investigationId);
    api
      .getReport(investigationId)
      .then(setReport)
      .catch(() => {
        /* no report yet — that's fine */
      });
  }, [investigationId]);

  async function runInvestigation() {
    if (!investigationId) return;
    setRunning(true);
    setError(null);
    try {
      const result = await api.investigate(investigationId, question || undefined);
      setReport(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Investigation Workspace"
        title={investigationId ? `Investigation ${investigationId.slice(0, 8)}` : "Loading investigation…"}
        description="Run hybrid retrieval + LLM reasoning across the ingested documents to identify a root cause, reconstruct a timeline, and generate a full RCA report."
      />

      <div className="grid grid-cols-3 gap-6 px-8 py-6">
        <div className="col-span-2 space-y-6">
          <Panel className="p-5">
            <label className="mb-2 block text-[12px] font-medium text-muted">
              Focusing question (optional)
            </label>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. Why did the payment service fail?"
              className="w-full rounded-md border border-line bg-panel2 px-3 py-2 text-[13px] text-ink outline-none placeholder:text-faint focus:border-signal"
            />
            <div className="mt-4 flex justify-end">
              <PrimaryButton onClick={runInvestigation} disabled={running}>
                {running ? "Investigating…" : report ? "Re-run investigation" : "Run investigation"}
              </PrimaryButton>
            </div>
            {error && <p className="mt-3 text-[13px] text-critical">{error}</p>}
          </Panel>

          {report && (
            <Panel className="p-5">
              <p className="mb-1 text-[11px] text-faint">Root cause</p>
              <p className="text-[15px] font-medium text-ink">{report.root_cause.root_cause}</p>
              <p className="mt-3 text-[13px] leading-relaxed text-muted">{report.executive_summary}</p>
              <div className="mt-4 max-w-xs">
                <ConfidenceBar value={report.confidence} />
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {report.root_cause.cause_chain.map((step, i) => (
                  <span key={i} className="flex items-center gap-2">
                    <span className="rounded border border-line bg-panel2 px-2 py-1 font-mono text-[11px] text-ink">
                      {step}
                    </span>
                    {i < report.root_cause.cause_chain.length - 1 && (
                      <span className="text-faint">→</span>
                    )}
                  </span>
                ))}
              </div>
            </Panel>
          )}
        </div>

        <div className="space-y-3">
          <WorkspaceLink
            href={`/graph/${investigationId}`}
            label="Knowledge Graph Viewer"
            hint="Explore entities and causal relationships"
            disabled={!investigationId}
          />
          <WorkspaceLink
            href={`/timeline/${investigationId}`}
            label="Timeline Viewer"
            hint="See the reconstructed incident sequence"
            disabled={!investigationId || !report}
          />
          <WorkspaceLink
            href={`/report/${investigationId}`}
            label="RCA Report"
            hint="Full executive-ready report"
            disabled={!investigationId || !report}
          />
          <WorkspaceLink
            href={`/chat/${investigationId}`}
            label="Investigation Chat"
            hint="Ask follow-up questions"
            disabled={!investigationId}
          />
        </div>
      </div>
    </div>
  );
}

function WorkspaceLink({
  href,
  label,
  hint,
  disabled,
}: {
  href: string;
  label: string;
  hint: string;
  disabled?: boolean;
}) {
  const content = (
    <Panel
      className={`p-4 transition-colors ${disabled ? "opacity-40" : "hover:border-signal/50"}`}
    >
      <p className="text-[13px] font-medium text-ink">{label}</p>
      <p className="mt-1 text-[11px] text-muted">{hint}</p>
    </Panel>
  );
  if (disabled) return content;
  return <Link href={href}>{content}</Link>;
}
