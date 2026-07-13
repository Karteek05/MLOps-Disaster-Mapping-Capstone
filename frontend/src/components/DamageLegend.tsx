import type { ClassBreakdown } from "@/lib/api";

interface DamageLegendProps {
  classes: ClassBreakdown[];
}

function formatClassName(name: string): string {
  return name
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

export default function DamageLegend({ classes }: DamageLegendProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
        {classes.map((c) => (
          <div
            key={c.id}
            style={{ width: `${c.percentage}%`, backgroundColor: c.color }}
            title={`${c.name}: ${c.percentage.toFixed(1)}%`}
          />
        ))}
      </div>
      <ul className="flex flex-col gap-2">
        {classes.map((c) => (
          <li key={c.id} className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <span
                className="h-3 w-3 shrink-0 rounded-sm"
                style={{ backgroundColor: c.color }}
              />
              <span className="text-slate-300">{formatClassName(c.name)}</span>
            </span>
            <span className="tabular-nums text-slate-400">
              {c.percentage.toFixed(1)}% ({c.pixel_count.toLocaleString()} px)
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
