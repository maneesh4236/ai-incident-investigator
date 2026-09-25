"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { getCurrentInvestigationId } from "@/lib/api";

/**
 * The sidebar links to routes like /graph/current before it knows the real
 * investigation id (it resolves this client-side after mount). This hook
 * catches that literal "current" segment and swaps in the real id from
 * localStorage, or bounces to the dashboard if none exists yet.
 */
export function useResolvedInvestigationId(rawId: string): string | null {
  const router = useRouter();
  const [resolved, setResolved] = useState<string | null>(rawId === "current" ? null : rawId);

  useEffect(() => {
    if (rawId !== "current") {
      setResolved(rawId);
      return;
    }
    const stored = getCurrentInvestigationId();
    if (stored) {
      setResolved(stored);
    } else {
      router.replace("/");
    }
  }, [rawId, router]);

  return resolved;
}
