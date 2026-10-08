"use client";

import { useEffect, useState } from "react";
import { api, clearCurrentInvestigationId, InvestigationStatusDto, isNotFound } from "@/lib/api";

/**
 * Investigation lifecycle as seen by the UI:
 *   loading    -> status request in flight
 *   not_found  -> the backend does not know this id (investigations are kept in
 *                 memory, so a backend restart clears them) - upload again
 *   no_report  -> investigation exists, report not generated yet - run it
 *   has_report -> GET /report will succeed
 *   error      -> backend unreachable / unexpected failure
 * Pages check this before requesting /report or /timeline, so a missing report
 * is an intentional state instead of a 404 treated as an error.
 */
export type LifecycleState = "loading" | "not_found" | "no_report" | "has_report" | "error";

export function useInvestigationStatus(investigationId: string | null) {
  const [state, setState] = useState<LifecycleState>("loading");
  const [status, setStatus] = useState<InvestigationStatusDto | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!investigationId) return;
    let cancelled = false;
    setState("loading");
    api
      .getInvestigation(investigationId)
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        setState(s.has_report ? "has_report" : "no_report");
      })
      .catch((e) => {
        if (cancelled) return;
        if (isNotFound(e)) {
          clearCurrentInvestigationId(investigationId);
          setState("not_found");
        } else {
          setError(e.message);
          setState("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [investigationId]);

  return { state, status, error };
}

export const NOT_FOUND_MESSAGE =
  "This investigation is no longer available on the server. Investigations are kept in memory, so a backend " +
  "restart clears them - upload the logs again to start a new investigation.";
