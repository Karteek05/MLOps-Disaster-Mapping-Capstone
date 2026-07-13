"use client";

import { useState } from "react";
import type { PredictResponse } from "@/lib/api";
import DamageLegend from "./DamageLegend";

interface ResultViewProps {
  result: PredictResponse;
  postImageUrl: string;
}

export default function ResultView({ result, postImageUrl }: ResultViewProps) {
  const [view, setView] = useState<"before" | "after">("after");

  return (
    <div className="flex flex-col gap-6 lg:flex-row">
      <div className="flex-1">
        <div className="mb-3 inline-flex rounded-lg bg-slate-800 p-1">
          <button
            type="button"
            onClick={() => setView("before")}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              view === "before" ? "bg-slate-600 text-white" : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Before
          </button>
          <button
            type="button"
            onClick={() => setView("after")}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              view === "after" ? "bg-slate-600 text-white" : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Damage Map
          </button>
        </div>
        <div className="relative aspect-square w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={postImageUrl}
            alt="Post-disaster"
            className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${
              view === "before" ? "opacity-100" : "opacity-0"
            }`}
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.overlay_image}
            alt="Damage assessment overlay"
            className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${
              view === "after" ? "opacity-100" : "opacity-0"
            }`}
          />
        </div>
      </div>
      <div className="w-full rounded-xl border border-slate-800 bg-slate-900/60 p-5 lg:w-80">
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Damage Breakdown
        </h3>
        <DamageLegend classes={result.classes} />
      </div>
    </div>
  );
}
