"use client";

import { use, useEffect, useState } from "react";
import { api, Timeline } from "@/lib/api";
import { useResolvedInvestigationId } from "@/lib/useInvestigationId";
import { EmptyState, PageHeader, Panel, Severity } from "@/components/ui";

export default function TimelinePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const investigationId = useResolvedInvestigationId(id);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!investigationId) return;
    api
      .getTimeline(investigationId)
      .then(setTimeline)
      .catch((e) => setError(e.message));
  }, [investigationId]);

  return (
    <div>
      <PageHeader
        eyebrow="Timeline"
        title="Incident Timeline Reconstruction"
        description="Chronological sequence of events correlated across logs and documents, ordered by extracted timestamps."
      />

      <div className="mx-auto max-w-3xl px-8 py-8">
        {error && (
          <Panel className="border-critical/40 p-5 text-[13px] text-critical">
            {error}. Run an investigation from the workspace first.
          </Panel>
        )}

        {timeline && timeline.events.length === 0 && (
          <EmptyState title="No events yet" description="Run an investigation to reconstruct the timeline." />
        )}

        {timeline && timeline.events.length > 0 && (
          <div className="relative pl-8">
            <div className="absolute bottom-0 left-[9px] top-2 w-px bg-line" />
            {timeline.events.map((event, i) => (
              <div key={i} className="relative mb-8 last:mb-0">
                <div
                  className="absolute -left-8 top-1 h-3.5 w-3.5 rounded-full border-2 border-base"
                  style={{
                    backgroundColor:
                      event.severity === "critical"
                        ? "#E5555C"
                        : event.severity === "warning"
                          ? "#F0A639"
                          : "#5A6675",
                  }}
                />
                <Panel className="p-4">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="font-mono text-[12px] text-signal">
                      {event.timestamp || `step ${i + 1}`}
                    </span>
                    <Severity level={event.severity} />
                  </div>
                  <p className="text-[13px] font-medium text-ink">{event.title}</p>
                  <p className="mt-1 text-[12px] leading-relaxed text-muted">{event.description}</p>
                </Panel>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
