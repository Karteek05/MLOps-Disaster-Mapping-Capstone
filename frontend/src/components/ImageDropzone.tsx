"use client";

import { useEffect, useMemo, useRef, useState } from "react";

interface ImageDropzoneProps {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
}

export default function ImageDropzone({ label, file, onChange }: ImageDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFiles = (files: FileList | null) => {
    const picked = files?.[0];
    if (picked && picked.type.startsWith("image/")) {
      onChange(picked);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-slate-300">{label}</span>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`relative flex h-48 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition-colors ${
          isDragOver
            ? "border-sky-400 bg-sky-950/40"
            : "border-slate-700 bg-slate-900/60 hover:border-slate-500"
        }`}
      >
        {previewUrl ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={previewUrl} alt={label} className="h-full w-full object-cover" />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onChange(null);
              }}
              className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-black/70 text-white hover:bg-black/90"
              aria-label={`Clear ${label}`}
            >
              ×
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center text-slate-400">
            <span className="text-3xl">📷</span>
            <span className="text-sm">Drag & drop or click to upload</span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}
