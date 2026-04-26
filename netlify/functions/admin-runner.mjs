function json(body, status = 200) {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function getToken(url) {
  return (new URL(url).searchParams.get("token") || "").trim();
}

function isAuthed(url) {
  const expected = (process.env.ADMIN_RUN_TOKEN || "").trim();
  const token = getToken(url);
  return Boolean(expected && token && token === expected);
}

async function hit(url, label) {
  const res = await fetch(url, {
    method: "GET",
    headers: { "content-type": "application/json" },
  });
  const text = await res.text();
  return {
    label,
    ok: res.ok,
    status: res.status,
    body: text,
  };
}

export default async (req) => {
  try {
    const url = new URL(req.url);

    if (!isAuthed(req.url)) {
      return json({ ok: false, error: "Unauthorized" }, 401);
    }

    const mode = (url.searchParams.get("mode") || "status").trim();
    const origin = `${url.protocol}//${url.host}`;

    const buildHook = (process.env.NETLIFY_BUILD_HOOK_URL || "").trim();

    if (mode === "status") {
      return json({
        ok: true,
        mode,
        has_build_hook: Boolean(buildHook),
        has_admin_token: Boolean((process.env.ADMIN_RUN_TOKEN || "").trim()),
        usage: {
          status: "/.netlify/functions/admin-runner?token=ADMIN_RUN_TOKEN",
          rebuild_only: "/.netlify/functions/admin-runner?mode=rebuild_only&token=ADMIN_RUN_TOKEN",
          aaa_ingest_status: "/.netlify/functions/admin-runner?mode=aaa_ingest_status&token=ADMIN_RUN_TOKEN",
          aaa_ingest_run_example: "/.netlify/functions/admin-runner?mode=aaa_ingest_run&week_start=2026-04-20&season=2026&max_teams=11&token=ADMIN_RUN_TOKEN",
          chain_example: "/.netlify/functions/admin-runner?mode=aaa_then_rebuild&week_start=2026-04-20&season=2026&max_teams=11&token=ADMIN_RUN_TOKEN"
        }
      });
    }

    if (mode === "rebuild_only") {
      if (!buildHook) {
        return json({ ok: false, error: "Missing NETLIFY_BUILD_HOOK_URL" }, 500);
      }

      const res = await fetch(buildHook, {
        method: "POST",
        headers: { "content-type": "application/json" },
      });
      const text = await res.text();

      return json({
        ok: res.ok,
        mode,
        status: res.status,
        response: text,
      }, res.ok ? 200 : 500);
    }

    if (mode === "aaa_ingest_status") {
      const target = `${origin}/.netlify/functions/ingest-milb-aaa-weekly?mode=status`;
      const result = await hit(target, "aaa_ingest_status");
      return json({ ok: result.ok, mode, result }, result.ok ? 200 : 500);
    }

    if (mode === "aaa_ingest_run" || mode === "aaa_then_rebuild") {
      const weekStart = (url.searchParams.get("week_start") || "").trim();
      const season = (url.searchParams.get("season") || "").trim();
      const maxTeams = (url.searchParams.get("max_teams") || "").trim();
      const startIndex = (url.searchParams.get("start_index") || "").trim();

      if (!weekStart || !season) {
        return json({ ok: false, error: "Missing week_start or season" }, 400);
      }

      const ingestUrl = new URL(`${origin}/.netlify/functions/ingest-milb-aaa-weekly`);
      ingestUrl.searchParams.set("week_start", weekStart);
      ingestUrl.searchParams.set("season", season);
      ingestUrl.searchParams.set("token", (process.env.ADMIN_RUN_TOKEN || "").trim());
      if (maxTeams) ingestUrl.searchParams.set("max_teams", maxTeams);
      if (startIndex) ingestUrl.searchParams.set("start_index", startIndex);

      const ingestResult = await hit(ingestUrl.toString(), "aaa_ingest_run");

      if (mode === "aaa_ingest_run") {
        return json({
          ok: ingestResult.ok,
          mode,
          ingest: ingestResult,
        }, ingestResult.ok ? 200 : 500);
      }

      if (!buildHook) {
        return json({
          ok: false,
          mode,
          ingest: ingestResult,
          error: "Missing NETLIFY_BUILD_HOOK_URL",
        }, 500);
      }

      const rebuildRes = await fetch(buildHook, {
        method: "POST",
        headers: { "content-type": "application/json" },
      });
      const rebuildText = await rebuildRes.text();

      return json({
        ok: ingestResult.ok && rebuildRes.ok,
        mode,
        ingest: ingestResult,
        rebuild: {
          ok: rebuildRes.ok,
          status: rebuildRes.status,
          body: rebuildText,
        },
      }, ingestResult.ok && rebuildRes.ok ? 200 : 500);
    }

    return json({ ok: false, error: `Unknown mode: ${mode}` }, 400);
  } catch (error) {
    return json({ ok: false, error: String(error) }, 500);
  }
};
