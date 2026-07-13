export type DamageClassId = 0 | 1 | 2 | 3 | 4;

export const DAMAGE_CLASS_ORDER: DamageClassId[] = [0, 1, 2, 3, 4];

export const DAMAGE_CLASS_LABELS: Record<DamageClassId, string> = {
  0: "Background",
  1: "No Damage",
  2: "Minor Damage",
  3: "Major Damage",
  4: "Destroyed",
};

// Mirrors backend/app/inference.py's COLOR_MAP.
export const DAMAGE_CLASS_COLORS: Record<DamageClassId, string> = {
  0: "#000000",
  1: "#00ff00",
  2: "#ffff00",
  3: "#ff8000",
  4: "#ff0000",
};
