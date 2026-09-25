"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getCurrentInvestigationId } from "@/lib/api";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", glyph: "◆" },
  { href: "/upload", label: "Upload Center", glyph: "↑" },
  { href: "/investigate/current", label: "Workspace", glyph: "▣" },
  { href: "/graph/current", label: "Knowledge Graph", glyph: "◈" },
  { href: "/timeline/current", label: "Timeline", glyph: "≣" },
  { href: "/report/current", label: "RCA Report", glyph: "▤" },
  { href: "/chat/current", label: "Investigation Chat", glyph: "◎" },
];

function resolveHref(href: string, currentId: string | null) {
  if (href.endsWith("/current") && currentId) {
    return href.replace("current", currentId);
  }
  return href;
}

export function Sidebar() {
  const pathname = usePathname();
  const [currentId, setCurrentId] = useState<string | null>(null);

  useEffect(() => {
    setCurrentId(getCurrentInvestigationId());
  }, [pathname]);

  return (
    <aside className="flex h-screen w-60 shrink-0 flex-col border-r border-line bg-panel">
      <div className="flex items-center gap-2 border-b border-line px-5 py-5">
        <span className="text-signal text-lg leading-none">◆</span>
        <div>
          <p className="text-[13px] font-semibold tracking-tight text-ink">AetherLog</p>
          <p className="text-[11px] text-faint">Incident Investigator</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4">
        {NAV_ITEMS.map((item) => {
          const href = resolveHref(item.href, currentId);
          const active =
            pathname === href || (item.href !== "/" && pathname.startsWith(href.split("/current")[0] + "/") && href !== item.href);
          const disabled = item.href.endsWith("/current") && !currentId;
          return (
            <Link
              key={item.href}
              href={disabled ? "#" : href}
              aria-disabled={disabled}
              className={`mb-1 flex items-center gap-3 rounded-md px-3 py-2 text-[13px] transition-colors ${
                active
                  ? "bg-panel2 text-ink"
                  : disabled
                    ? "cursor-not-allowed text-faint"
                    : "text-muted hover:bg-panel2 hover:text-ink"
              }`}
            >
              <span className="w-4 text-center text-signal/80">{item.glyph}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-line px-5 py-4">
        <p className="text-[11px] leading-relaxed text-faint">
          Hybrid retrieval over Qdrant + Neo4j, reasoned by Gemini 2.5 Flash.
        </p>
      </div>
    </aside>
  );
}
