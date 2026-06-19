function json(body, status = 200) {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function getTokenFromRequest(req, url) {
  const headerToken =
    req.headers.get("x-admin-token") ||
    req.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ||
    "";

  const queryToken = url.searchParams.get("token") || "";
  return String(headerToken || queryToken || "").trim();
}

function isAuthed(req, url) {
  const expected = String(process.env.ADMIN_RUN_TOKEN || "").trim();
  const token = getTokenFromRequest(req, url);
  return Boolean(expected && token && token === expected);
}

function isUuid(value) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    String(value || "").trim()
  );
}

export default async (req) => {
  try {
    const url = new URL(req.url);

    if (!isAuthed(req, url)) {
      return json({ ok: false, error: "Unauthorized" }, 401);
    }

    if (req.method !== "POST") {
      return json({ ok: false, error: "Method not allowed" }, 405);
    }

    const body = await req.json().catch(() => ({}));
    const jobId = String(body.job_id || body.jobId || url.searchParams.get("job_id") || "").trim();

    if (!isUuid(jobId)) {
      return json({ ok: false, error: "Invalid or missing job_id" }, 400);
    }

    const origin = `${url.protocol}//${url.host}`;
    const workerUrl = `${origin}/.netlify/functions/admin-audit-worker-background?token=${encodeURIComponent(
      String(process.env.ADMIN_RUN_TOKEN || "").trim()
    )}`;

    const workerResponse = await fetch(workerUrl, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        job_id: jobId,
        action: "green_baseline",
        requested_at: new Date().toISOString(),
      }),
    });

    return json(
      {
        ok: workerResponse.ok,
        accepted: workerResponse.ok,
        job_id: jobId,
        worker_status: workerResponse.status,
        message: workerResponse.ok
          ? "Green baseline background audit runner accepted the job."
          : "Background audit runner did not accept the job.",
      },
      workerResponse.ok ? 200 : 500
    );
  } catch (error) {
    return json({ ok: false, error: String(error?.message || error) }, 500);
  }
};
