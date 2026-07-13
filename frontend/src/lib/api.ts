export interface ClassBreakdown {
  id: number;
  name: string;
  color: string;
  pixel_count: number;
  percentage: number;
}

export interface PredictResponse {
  overlay_image: string;
  img_size: number;
  classes: ClassBreakdown[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function predictDamage(pre: File, post: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("pre_disaster_image", pre);
  formData.append("post_disaster_image", post);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}) as { detail?: string });
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }

  return res.json();
}
