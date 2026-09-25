"use client";

import { useRouter } from "next/navigation";
import { ChangeEvent, DragEvent, useState } from "react";
import { api, setCurrentInvestigationId } from "@/lib/api";
import { PageHeader, Panel, PrimaryButton } from "@/components/ui";

export default function UploadPage() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [title, setTitle] = useState("");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function addFiles(list: FileList | null) {
    if (!list) return;
    setFiles((prev) => [...prev, ...Array.from(list)]);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleSubmit() {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const result = await api.uploadDocuments(files, undefined, title || undefined);
      setCurrentInvestigationId(result.investigation_id);
      router.push(`/investigate/${result.investigation_id}`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Ingestion"
        title="Upload Center"
        description="Drop in logs, PDFs, incident reports, runbooks, or architecture docs. Each file is chunked, embedded, and mined for entities to build the knowledge graph."
      />

      <div className="mx-auto max-w-2xl px-8 py-8">
        <label className="mb-2 block text-[12px] font-medium text-muted">
          Investigation title (optional)
        </label>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Checkout outage — May 1"
          className="mb-6 w-full rounded-md border border-line bg-panel px-3 py-2 text-[13px] text-ink outline-none placeholder:text-faint focus:border-signal"
        />

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-8 py-14 text-center transition-colors ${
            dragging ? "border-signal bg-signal/5" : "border-line bg-panel"
          }`}
        >
          <p className="text-[13px] text-ink">Drag files here, or</p>
          <label className="mt-3 cursor-pointer rounded-md border border-line px-4 py-2 text-[13px] text-ink hover:bg-panel2">
            Browse files
            <input
              type="file"
              multiple
              className="hidden"
              accept=".log,.txt,.pdf,.md"
              onChange={(e: ChangeEvent<HTMLInputElement>) => addFiles(e.target.files)}
            />
          </label>
          <p className="mt-3 text-[11px] text-faint">.log · .txt · .pdf · .md — up to 50MB each</p>
        </div>

        {files.length > 0 && (
          <Panel className="mt-6 divide-y divide-line">
            {files.map((f, i) => (
              <div key={`${f.name}-${i}`} className="flex items-center justify-between px-4 py-3 text-[13px]">
                <div>
                  <p className="text-ink">{f.name}</p>
                  <p className="font-mono text-[11px] text-faint">{(f.size / 1024).toFixed(1)} KB</p>
                </div>
                <button onClick={() => removeFile(i)} className="text-faint hover:text-critical">
                  Remove
                </button>
              </div>
            ))}
          </Panel>
        )}

        {error && (
          <p className="mt-4 text-[13px] text-critical">
            {error}. Confirm the backend is reachable at NEXT_PUBLIC_API_BASE_URL.
          </p>
        )}

        <div className="mt-6 flex justify-end">
          <PrimaryButton onClick={handleSubmit} disabled={files.length === 0 || uploading}>
            {uploading ? "Processing…" : `Ingest ${files.length || ""} document${files.length === 1 ? "" : "s"}`}
          </PrimaryButton>
        </div>
      </div>
    </div>
  );
}
