"use client";

import { use, useEffect, useMemo, useState } from "react";
import { api, KnowledgeGraphResponse } from "@/lib/api";
import { useResolvedInvestigationId } from "@/lib/useInvestigationId";
import { EmptyState, PageHeader, Panel } from "@/components/ui";

const TYPE_COLORS: Record<string, string> = {
  SERVICE: "#31C6AD",
  ERROR: "#E5555C",
  COMPONENT: "#F0A639",
  DATABASE: "#7C9CF0",
  API: "#C77DE0",
  EVENT: "#8B98A8",
  SYSTEM: "#5AC8FA",
};

export default function GraphViewerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const investigationId = useResolvedInvestigationId(id);
  const [graph, setGraph] = useState<KnowledgeGraphResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!investigationId) return;
    api
      .getGraph(investigationId)
      .then(setGraph)
      .catch((e) => setError(e.message));
  }, [investigationId]);

  const layout = useMemo(() => {
    if (!graph || graph.nodes.length === 0) return null;
    const size = 560;
    const center = size / 2;
    const radius = size / 2 - 60;
    const positions: Record<string, { x: number; y: number }> = {};
    graph.nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / graph.nodes.length - Math.PI / 2;
      positions[node.name] = {
        x: center + radius * Math.cos(angle),
        y: center + radius * Math.sin(angle),
      };
    });
    return { size, positions };
  }, [graph]);

  return (
    <div>
      <PageHeader
        eyebrow="Knowledge Graph"
        title="Entity & Causal Relationship Viewer"
        description="Services, errors, components, and events extracted from the ingested documents, connected by CAUSES / DEPENDS_ON / TRIGGERS / CONTAINS / RELATED_TO edges."
      />

      <div className="grid grid-cols-3 gap-6 px-8 py-6">
        <Panel className="col-span-2 flex items-center justify-center p-6">
          {error && <p className="text-[13px] text-critical">{error}</p>}
          {!error && graph && graph.nodes.length === 0 && (
            <EmptyState
              title="No graph yet"
              description="Upload documents and run an investigation to populate the knowledge graph."
            />
          )}
          {layout && graph && (
            <svg width={layout.size} height={layout.size} className="max-w-full">
              <defs>
                <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 Z" fill="#3A4756" />
                </marker>
              </defs>
              {graph.edges.map((edge) => {
                const from = layout.positions[edge.source];
                const to = layout.positions[edge.target];
                if (!from || !to) return null;
                return (
                  <line
                    key={edge.id}
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={selected && (edge.source === selected || edge.target === selected) ? "#F0A639" : "#2A3746"}
                    strokeWidth={selected && (edge.source === selected || edge.target === selected) ? 1.6 : 1}
                    markerEnd="url(#arrow)"
                  />
                );
              })}
              {graph.nodes.map((node) => {
                const pos = layout.positions[node.name];
                if (!pos) return null;
                const isSelected = selected === node.name;
                return (
                  <g
                    key={node.id}
                    transform={`translate(${pos.x}, ${pos.y})`}
                    onMouseEnter={() => setSelected(node.name)}
                    onMouseLeave={() => setSelected(null)}
                    className="cursor-pointer"
                  >
                    <circle
                      r={isSelected ? 8 : 6}
                      fill={TYPE_COLORS[node.type] || "#8B98A8"}
                      stroke="#0A0E13"
                      strokeWidth={2}
                    />
                    <text
                      x={0}
                      y={-12}
                      textAnchor="middle"
                      className="font-mono"
                      fontSize={10}
                      fill={isSelected ? "#E8EDF2" : "#8B98A8"}
                    >
                      {node.name}
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </Panel>

        <div className="space-y-4">
          <Panel className="p-4">
            <p className="mb-3 text-[11px] font-medium text-faint">Entity types</p>
            <div className="space-y-2">
              {Object.entries(TYPE_COLORS).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2 text-[12px] text-muted">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                  {type}
                </div>
              ))}
            </div>
          </Panel>

          {graph && (
            <Panel className="max-h-[360px] overflow-y-auto p-4 scrollbar-thin">
              <p className="mb-3 text-[11px] font-medium text-faint">
                Relationships ({graph.edges.length})
              </p>
              <div className="space-y-2 font-mono text-[11px] text-muted">
                {graph.edges.map((edge) => (
                  <div key={edge.id} className="rounded border border-line bg-panel2 px-2 py-1.5">
                    <span className="text-ink">{edge.source}</span>{" "}
                    <span className="text-signal">{edge.type}</span>{" "}
                    <span className="text-ink">{edge.target}</span>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
