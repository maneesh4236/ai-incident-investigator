"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, InvestigationSummary, setCurrentInvestigationId } from "@/lib/api";
import { EmptyState, PageHeader, Panel, PrimaryButton, StatusPill } from "@/components/ui";

export default function DashboardPage() {
  const [investigations, setInvestigations] = useState<InvestigationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listInvestigations()
      .then(setInvestigations)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <PageHeader
        eyebrow="Overview"
        title="Investigation Dashboard"
        description="Every incident investigation currently held in memory, with ingestion status and quick access to its workspace."
        action={
          <Link href="/upload">
            <PrimaryButton>New investigation</PrimaryButton>
          </Link>
        }
      />

      <div className="grid grid-cols-3 gap-4 px-8 py-6">
        <Panel className="px-5 py-4">
          <p className="text-[11px] text-faint">Total investigations</p>
          <p className="mt-2 font-mono text-2xl text-ink">{investigations.length}</p>
        </Panel>
        <Panel className="px-5 py-4">
          <p className="text-[11px] text-faint">Completed RCAs</p>
          <p className="mt-2 font-mono text-2xl text-trace">
            {investigations.filter((i) => i.status === "COMPLETED").length}
          </p>
        </Panel>
        <Panel className="px-5 py-4">
          <p className="text-[11px] text-faint">Documents ingested</p>
          <p className="mt-2 font-mono text-2xl text-ink">
            {investigations.reduce((sum, i) => sum + i.document_count, 0)}
          </p>
        </Panel>
      </div>

      <div className="px-8 pb-10">
        {loading && <p className="text-[13px] text-muted">Loading investigations…</p>}
        {error && (
          <Panel className="border-critical/40 px-5 py-4 text-[13px] text-critical">
            Could not reach the backend ({error}). Is the FastAPI service running?
          </Panel>
        )}
        {!loading && !error && investigations.length === 0 && (
          <EmptyState
            title="No investigations yet"
            description="Upload logs, incident reports, or runbooks to start your first AI-driven root cause investigation."
            action={
              <Link href="/upload">
                <PrimaryButton>Upload documents</PrimaryButton>
              </Link>
            }
          />
        )}
        {!loading && investigations.length > 0 && (
          <Panel>
            <table className="w-full text-left text-[13px]">
              <thead>
                <tr className="border-b border-line text-[11px] text-faint">
                  <th className="px-5 py-3 font-medium">Investigation</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                  <th className="px-5 py-3 font-medium">Documents</th>
                  <th className="px-5 py-3 font-medium">Created</th>
                  <th className="px-5 py-3" />
                </tr>
              </thead>
              <tbody>
                {investigations.map((inv) => (
                  <tr key={inv.id} className="border-b border-line last:border-0">
                    <td className="px-5 py-3 text-ink">{inv.title}</td>
                    <td className="px-5 py-3">
                      <StatusPill status={inv.status} />
                    </td>
                    <td className="px-5 py-3 font-mono text-muted">{inv.document_count}</td>
                    <td className="px-5 py-3 text-muted">
                      {new Date(inv.created_at).toLocaleString()}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button
                        onClick={() => setCurrentInvestigationId(inv.id)}
                        className="text-trace hover:underline"
                      >
                        <Link href={`/investigate/${inv.id}`}>Open workspace →</Link>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </div>
    </div>
  );
}
