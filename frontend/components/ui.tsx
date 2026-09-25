import { HTMLAttributes, ReactNode } from "react";

export function Panel({
  children,
  className = "",
  ...rest
}: { children: ReactNode } & HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`rounded-lg border border-line bg-panel shadow-panel ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between border-b border-line px-8 py-6">
      <div>
        {eyebrow && <p className="mb-1 text-[11px] text-faint">{eyebrow}</p>}
        <h1 className="text-xl font-semibold text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-[13px] text-muted">{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function StatusPill({ status }: { status: string }) {
  const styles: Record<string, string> = {
    PENDING: "bg-panel2 text-muted",
    PROCESSING: "bg-signal/15 text-signal",
    COMPLETED: "bg-trace/15 text-trace",
    FAILED: "bg-critical/15 text-critical",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${
        styles[status] || styles.PENDING
      }`}
    >
      <span
        className={`status-dot ${status === "PROCESSING" ? "animate-pulse-signal" : ""}`}
        style={{
          backgroundColor:
            status === "COMPLETED"
              ? "#31C6AD"
              : status === "FAILED"
                ? "#E5555C"
                : status === "PROCESSING"
                  ? "#F0A639"
                  : "#5A6675",
        }}
      />
      {status}
    </span>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "bg-trace" : pct >= 40 ? "bg-signal" : "bg-critical";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] text-muted">
        <span>Confidence</span>
        <span className="font-mono text-ink">{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel2">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Severity({ level }: { level: string }) {
  const map: Record<string, string> = {
    critical: "text-critical",
    warning: "text-signal",
    info: "text-muted",
  };
  const dot: Record<string, string> = {
    critical: "#E5555C",
    warning: "#F0A639",
    info: "#5A6675",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-medium ${map[level] || map.info}`}>
      <span className="status-dot" style={{ backgroundColor: dot[level] || dot.info }} />
      {level}
    </span>
  );
}

export function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-line px-8 py-16 text-center">
      <p className="text-[14px] font-medium text-ink">{title}</p>
      <p className="mt-1 max-w-sm text-[13px] text-muted">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function PrimaryButton({
  children,
  ...rest
}: { children: ReactNode } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={`rounded-md bg-signal px-4 py-2 text-[13px] font-medium text-base transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 ${rest.className || ""}`}
    >
      {children}
    </button>
  );
}

export function GhostButton({
  children,
  ...rest
}: { children: ReactNode } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={`rounded-md border border-line px-4 py-2 text-[13px] font-medium text-ink transition-colors hover:bg-panel2 disabled:cursor-not-allowed disabled:opacity-40 ${rest.className || ""}`}
    >
      {children}
    </button>
  );
}
