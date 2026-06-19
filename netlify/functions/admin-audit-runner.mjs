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

function mustEnv(name) {
  const value = String(process.env[name] || "").trim();
  if (!value) throw new Error(`Missing env var: ${name}`);
  return value;
}

async function updateAuditJob(jobId, patch) {
  const supabaseUrl = mustEnv("SUPABASE_URL").replace(/\/+$/, "");
  const supabaseKey = mustEnv("SUPABASE_SERVICE_ROLE_KEY");

  const response = await fetch(`${supabaseUrl}/rest/v1/admin_audit_jobs?id=eq.${encodeURIComponent(jobId)}`, {
    method: "PATCH",
    headers: {
      apikey: supabaseKey,
      authorization: `Bearer ${supabaseKey}`,
      "content-type": "application/json",
      prefer: "return=representation",
    },
    body: JSON.stringify({
      ...patch,
      updated_at: new Date().toISOString(),
    }),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(`Supabase audit job update failed: HTTP ${response.status} ${JSON.stringify(data)}`);
  }

  return data;
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

    const buildHook =
      String(process.env.NETLIFY_GREEN_BASELINE_AUDIT_BUILD_HOOK_URL || "").trim() ||
      String(process.env.NETLIFY_BUILD_HOOK_URL || "").trim();

    if (!buildHook) {
      await updateAuditJob(jobId, {
        status: "failed",
        completed_at: new Date().toISOString(),
        message: "Green baseline audit runner could not start because the Netlify build hook is not configured.",
        error: "Missing NETLIFY_GREEN_BASELINE_AUDIT_BUILD_HOOK_URL or NETLIFY_BUILD_HOOK_URL.",
      });

      return json({
        ok: false,
        job_id: jobId,
        error: "Missing NETLIFY_GREEN_BASELINE_AUDIT_BUILD_HOOK_URL or NETLIFY_BUILD_HOOK_URL.",
      }, 500);
    }

    await updateAuditJob(jobId, {
      status: "queued",
      message: "Green baseline audit accepted. Netlify build hook is being triggered.",
      result_payload: {
        runner: {
          surface: "signals_netlify_function_to_build_hook",
          action: "green_baseline",
          accepted_at: new Date().toISOString(),
        },
      },
    });

    const buildResponse = await fetch(buildHook, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        job_id: jobId,
        action: "green_baseline",
        source: "admin_audit_runner",
        requested_at: new Date().toISOString(),
      }),
    });

    const buildBody = await buildResponse.text().catch(() => "");

    if (!buildResponse.ok) {
      await updateAuditJob(jobId, {
        status: "failed",
        completed_at: new Date().toISOString(),
        message: "Failed to trigger Netlify build hook for green baseline audit.",
        error: `Build hook HTTP ${buildResponse.status}: ${buildBody.slice(0, 500)}`,
      });
    } else {
      await updateAuditJob(jobId, {
        status: "queued",
        message: "Green baseline audit build hook accepted. Waiting for Netlify build runtime.",
        result_payload: {
          runner: {
            surface: "signals_netlify_function_to_build_hook",
            action: "green_baseline",
            build_hook_status: buildResponse.status,
            accepted_at: new Date().toISOString(),
          },
        },
      });
    }

    return json(
      {
        ok: buildResponse.ok,
        accepted: buildResponse.ok,
        job_id: jobId,
        build_hook_status: buildResponse.status,
        message: buildResponse.ok
          ? "Green baseline audit build hook accepted the job."
          : "Green baseline audit build hook failed to accept the job.",
      },
      buildResponse.ok ? 200 : 500
    );
  } catch (error) {
    return json({ ok: false, error: String(error?.message || error) }, 500);
  }
};
