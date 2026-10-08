"use client";

import { FormEvent, use, useRef, useState } from "react";
import { api, ChatMessageDto } from "@/lib/api";
import { useResolvedInvestigationId } from "@/lib/useInvestigationId";
import { PageHeader, Panel, PrimaryButton } from "@/components/ui";

const SUGGESTED_QUESTIONS = [
  "Why did the outage happen?",
  "What evidence supports this?",
  "Which service failed first?",
  "What was the impact?",
  "Show related incidents.",
];

type DisplayMessage = ChatMessageDto & { fallback?: boolean };

export default function ChatPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const investigationId = useResolvedInvestigationId(id);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEntities, setLastEntities] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function send(message: string) {
    if (!message.trim() || sending || !investigationId) return;
    const nextHistory: DisplayMessage[] = [...messages, { role: "user", content: message }];
    setMessages(nextHistory);
    setInput("");
    setSending(true);
    setError(null);
    try {
      const history: ChatMessageDto[] = messages.map(({ role, content }) => ({ role, content }));
      const response = await api.chat(investigationId, message, history);
      setMessages([
        ...nextHistory,
        {
          role: "assistant",
          content: response.answer,
          fallback: response.reasoning_mode === "deterministic_fallback",
        },
      ]);
      setLastEntities(response.referenced_entities);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSending(false);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  return (
    <div className="flex h-screen flex-col">
      <PageHeader
        eyebrow="Conversational Investigation"
        title="Investigation Chat"
        description="Ask follow-up questions grounded in retrieved evidence and the knowledge graph."
      />

      <div className="flex flex-1 flex-col overflow-hidden px-8 py-6">
        <div className="flex-1 space-y-4 overflow-y-auto scrollbar-thin pr-2">
          {messages.length === 0 && (
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="rounded-full border border-line px-3 py-1.5 text-[12px] text-muted hover:border-signal hover:text-ink"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-xl whitespace-pre-line rounded-lg px-4 py-3 text-[13px] leading-relaxed ${
                  m.role === "user" ? "bg-signal/15 text-ink" : "border border-line bg-panel text-ink"
                }`}
              >
                {m.fallback && (
                  <p className="mb-2 text-[11px] font-medium text-faint">
                    Deterministic fallback (Gemini unavailable) — answered from log evidence
                  </p>
                )}
                {m.content}
              </div>
            </div>
          ))}

          {sending && (
            <div className="flex justify-start">
              <div className="rounded-lg border border-line bg-panel px-4 py-3 text-[13px] text-muted">
                Investigating…
              </div>
            </div>
          )}

          {error && <p className="text-[13px] text-critical">{error}</p>}
          <div ref={bottomRef} />
        </div>

        {lastEntities.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2 border-t border-line pt-3">
            <span className="text-[11px] text-faint">Referenced:</span>
            {lastEntities.map((e, i) => (
              <span key={i} className="rounded bg-panel2 px-2 py-0.5 font-mono text-[11px] text-trace">
                {e}
              </span>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-3 border-t border-line pt-4">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about this investigation…"
            className="flex-1 rounded-md border border-line bg-panel px-3 py-2 text-[13px] text-ink outline-none placeholder:text-faint focus:border-signal"
          />
          <PrimaryButton type="submit" disabled={sending || !input.trim()}>
            Send
          </PrimaryButton>
        </form>
      </div>
    </div>
  );
}
