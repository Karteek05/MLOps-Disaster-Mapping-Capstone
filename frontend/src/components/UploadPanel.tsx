"use client";

import ImageDropzone from "./ImageDropzone";

interface UploadPanelProps {
  preFile: File | null;
  postFile: File | null;
  onPreChange: (file: File | null) => void;
  onPostChange: (file: File | null) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export default function UploadPanel({
  preFile,
  postFile,
  onPreChange,
  onPostChange,
  onSubmit,
  isLoading,
}: UploadPanelProps) {
  const canSubmit = preFile !== null && postFile !== null && !isLoading;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <ImageDropzone label="Pre-Disaster Image" file={preFile} onChange={onPreChange} />
        <ImageDropzone label="Post-Disaster Image" file={postFile} onChange={onPostChange} />
      </div>
      <button
        type="button"
        onClick={onSubmit}
        disabled={!canSubmit}
        className="flex items-center justify-center gap-2 rounded-lg bg-sky-500 px-6 py-3 font-medium text-white transition-colors hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
      >
        {isLoading ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
            Running damage assessment...
          </>
        ) : (
          "Run damage assessment"
        )}
      </button>
    </div>
  );
}
