export const config = {
  schedule: "0 14 * * *"
};

export default async () => {
  const buildHook = process.env.NETLIFY_BUILD_HOOK_URL;

  if (!buildHook) {
    return new Response(
      JSON.stringify({ ok: false, error: "Missing NETLIFY_BUILD_HOOK_URL" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  const res = await fetch(buildHook, {
    method: "POST",
    headers: { "content-type": "application/json" }
  });

  const text = await res.text();

  return new Response(
    JSON.stringify({
      ok: res.ok,
      status: res.status,
      response: text
    }),
    { status: res.ok ? 200 : 500, headers: { "content-type": "application/json" } }
  );
};