import { spawn } from "node:child_process";

const MAX_LOG_CHARS = 120000;

function cleanText(value) {
  return String(value || "").slice(-MAX_LOG_CHARS);
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

function runGreenBaselineAudit() {
  return new Promise((resolve) => {
    const startedAt = Date.now();
    const child = spawn("bash", ["scripts/run_green_baseline_audit.sh"], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        CI: "1",
      },
      shell: false,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout = cleanText(stdout + chunk.toString());
    });

    child.stderr.on("data", (chunk) => {
      stderr = cleanText(stderr + chunk.toString());
    });

    child.on("error", (error) => {
      resolve({
        ok: false,
        exit_code: null,
        duration_ms: Date.now() - startedAt,
        stdout,
        stderr,
        error: String(error?.message || error),
      });
    });

    child.on("close", (code) => {
      resolve({
        ok: code === 0,
        exit_code: code,
        duration_ms: Date.now() - startedAt,
        stdout,
        stderr,
        error: code === 0 ? null : `Green baseline audit exited with code ${code}.`,
      });
    });
  });
}

export default async (req) => {
  const url = new URL(req.url);
  let jobId = "";

  try {
    if (!isAuthed(req, url)) {
      return;
    }

    const body = await req.json().catch(() => ({}));
    jobId = String(body.job_id || body.jobId || url.searchParams.get("job_id") || "").trim();

    if (!isUuid(jobId)) {
      return;
    }

    await updateAuditJob(jobId, {
      status: "running",
      started_at: new Date().toISOString(),
      message: "Green baseline audit runner started.",
      result_payload: {
        runner: {
          surface: "signals_netlify_background_function",
          action: "green_baseline",
          started_at: new Date().toISOString(),
        },
      },
    });

    const result = await runGreenBaselineAudit();

    await updateAuditJob(jobId, {
      status: result.ok ? "succeeded" : "failed",
      completed_at: new Date().toISOString(),
      message: result.ok
        ? "Green baseline audit passed."
        : "Green baseline audit failed. Review stdout/stderr payload.",
      error: result.error,
      result_payload: {
        runner: {
          surface: "signals_netlify_background_function",
          action: "green_baseline",
          completed_at: new Date().toISOString(),
          ok: result.ok,
          exit_code: result.exit_code,
          duration_ms: result.duration_ms,
        },
        stdout_tail: result.stdout,
        stderr_tail: result.stderr,
      },
    });
  } catch (error) {
    if (jobId && isUuid(jobId)) {
      try {
        await updateAuditJob(jobId, {
          status: "failed",
          completed_at: new Date().toISOString(),
          message: "Green baseline audit runner failed before completion.",
          error: String(error?.message || error),
        });
      } catch (_) {
        // Preserve the original failure in function logs.
      }
    }

    console.error("admin-audit-worker-background failed", error);
  }
};
