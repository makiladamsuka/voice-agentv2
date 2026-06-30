/**
 * Robot backend (v5 media server on :8080).
 *
 * When the Next.js app runs on a laptop and the robot runs on the Pi, set in
 * `.env.local`:
 *   NEXT_PUBLIC_ROBOT_API_URL=http://<pi-ip>:8080
 *   BACKEND_URL=http://<pi-ip>:8080
 *
 * NEXT_PUBLIC_* is used for browser-direct calls (eye color). BACKEND_URL is
 * for Next.js server-side API rewrites.
 */
export function getRobotApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_ROBOT_API_URL?.trim().replace(/\/$/, '');
  return configured || '';
}

export function robotApiUrl(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  const base = getRobotApiBase();
  return base ? `${base}${normalized}` : normalized;
}
