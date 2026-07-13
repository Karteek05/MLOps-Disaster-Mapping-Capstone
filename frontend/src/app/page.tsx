"use client";

import { useEffect, useMemo, useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import ResultView from "@/components/ResultView";
import { predictDamage, type PredictResponse } from "@/lib/api";

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [preFile, setPreFile] = useState<File | null>(null);
  const [postFile, setPostFile] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const postImageUrl = useMemo(() => (postFile ? URL.createObjectURL(postFile) : null), [postFile]);
  useEffect(() => {
    return () => {
      if (postImageUrl) URL.revokeObjectURL(postImageUrl);
    };
  }, [postImageUrl]);

  const handleSubmit = async () => {
    if (!preFile || !postFile) return;
    setStatus("loading");
    setError(null);
    try {
      const response = await predictDamage(preFile, postFile);
      setResult(response);
      setStatus("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("error");
    }
  };

  return (
    <div className="min-h-screen flex-1 bg-slate-950 text-slate-100">
      <main className="mx-auto flex max-w-4xl flex-col gap-8 px-6 py-12">
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold tracking-tight">Disaster Damage Mapping</h1>
          <p className="max-w-2xl text-slate-400">
            Upload a pre-disaster and post-disaster satellite image pair of the same area to get
            a building damage assessment map.
          </p>
        </header>

        <UploadPanel
          preFile={preFile}
          postFile={postFile}
          onPreChange={setPreFile}
          onPostChange={setPostFile}
          onSubmit={handleSubmit}
          isLoading={status === "loading"}
        />

        {status === "error" && error && (
          <div className="rounded-lg border border-red-900 bg-red-950/50 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {status === "success" && result && postImageUrl && (
          <ResultView result={result} postImageUrl={postImageUrl} />
        )}
      </main>
    </div>
  );
}
