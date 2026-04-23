/*
  ingest-milb-aaa-weekly.mjs
  Real AAA weekly ingest:
  - Pull roster (identity/positions)
  - Pull TEAM-FILTERED weekly stats splits (hitting + pitching) via /api/v1/stats
  - Map stats -> players
  - Upsert into public.milb_raw_weekly

  Required query params:
    - week_start (YYYY-MM-DD)
    - season (YYYY)

  Optional:
    - team_id (single-team mode; if omitted -> multi-org mode)
    - max_teams (multi-org mode limiter; default = all selected)
    - mode=status | run (default run)
    - token=ADMIN_RUN_TOKEN (required for run)

  Optional env vars:
    - MILB_ORG_ALLOWLIST (comma-separated MLB org names)
*/

export const config = {
  // schedule: "10 11 * * 1", // optional
};

/* ---------------- helpers ---------------- */

function mustEnv(name) {
  const v = (process.env[name] || "").trim();
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

function isAuthedFromUrl(urlStr) {
  const url = new URL(urlStr);
  const token = (url.searchParams.get("token") || "").trim();
  const expected = (process.env.ADMIN_RUN_TOKEN || "").trim();
  return Boolean(expected && token && token === expected);
}

function toISODate(d) {
  const dt = new Date(d);
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const day = String(dt.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function addDays(iso, days) {
  const dt = new Date(`${iso}T00:00:00Z`);
  dt.setUTCDate(dt.getUTCDate() + days);
  return toISODate(dt);
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function safeStr(x) {
  const s = String(x ?? "").trim();
  return s ? s : null;
}

async function fetchJson(url) {
  const res = await fetch(url, {
    headers: {
      "user-agent": "DiamondSignals/1.0 (AAA Ingest)",
      accept: "application/json",
    },
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`fetch ${res.status} ${res.statusText} url=${url} body=${txt.slice(0, 260)}`);
  }
  return res.json();
}

function getSplits(payload) {
  const splits = payload?.stats?.[0]?.splits;
  return Array.isArray(splits) ? splits : [];
}

/* ---------------- org allowlist ---------------- */

const DEFAULT_ORG_ALLOWLIST = [
  "Detroit Tigers",
  "New York Yankees",
  "Los Angeles Dodgers",
  "Atlanta Braves",
  "Chicago Cubs",
  "Houston Astros",
  "Boston Red Sox",
  "San Francisco Giants",
  "Texas Rangers",
  "Seattle Mariners",
  "New York Mets",
];

function getOrgAllowlist() {
  const raw = (process.env.MILB_ORG_ALLOWLIST || "").trim();
  if (!raw) return DEFAULT_ORG_ALLOWLIST.slice();
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

/* ---------------- MLB Stats API builders ---------------- */

function buildStatsUrl({ group, season, teamId, sportId, startDate, endDate }) {
  const base = "https://statsapi.mlb.com/api/v1/stats";
  const u = new URL(base);

  // critical: byDateRange + sportId=11 + teamId
  u.searchParams.set("stats", "byDateRange");
  u.searchParams.set("group", group); // hitting | pitching
  u.searchParams.set("season", String(season));
  u.searchParams.set("sportId", String(sportId));
  u.searchParams.set("teamId", String(teamId));
  u.searchParams.set("startDate", startDate);
  u.searchParams.set("endDate", endDate);

  return u.toString();
}

/* ---------------- team discovery (AAA teams by org) ---------------- */

async function fetchAaaTeamsForSeason(season) {
  // sportId=11 is AAA
  // This returns AAA clubs; each has parentOrgName (MLB org).
  const url = `https://statsapi.mlb.com/api/v1/teams?sportId=11&season=${encodeURIComponent(String(season))}`;
  const payload = await fetchJson(url);
  const teams = Array.isArray(payload?.teams) ? payload.teams : [];
  return teams
    .map((t) => ({
      teamId: num(t?.id),
      name: safeStr(t?.name),
      parentOrgName: safeStr(t?.parentOrgName),
      parentOrgId: num(t?.parentOrgId),
    }))
    .filter((t) => t.teamId && t.name);
}

function filterTeamsByOrgAllowlist(teams, allowlist) {
  const allow = new Set(allowlist.map((s) => s.toLowerCase()));
  return teams.filter((t) => {
    const org = (t.parentOrgName || "").toLowerCase();
    return allow.has(org);
  });
}

/* ---------------- ingest: single team core ---------------- */

async function ingestOneTeam({ sb, teamId, weekStart, weekEnd, season, sportId }) {
  // Team meta
  const teamMeta = await fetchJson(`https://statsapi.mlb.com/api/v1/teams/${teamId}`);
  const team = teamMeta?.teams?.[0] || {};
  const teamName = safeStr(team?.name) || `team_${teamId}`;
  const parentOrgName = safeStr(team?.parentOrgName) || null;

  // Roster (identity/pos)
  const rosterJson = await fetchJson(`https://statsapi.mlb.com/api/v1/teams/${teamId}/roster?season=${season}`);
  const roster = Array.isArray(rosterJson?.roster) ? rosterJson.roster : [];

  // Pull weekly team-filtered stats splits
  const hitUrl = buildStatsUrl({
    group: "hitting",
    season,
    teamId,
    sportId,
    startDate: weekStart,
    endDate: weekEnd,
  });

  const pitUrl = buildStatsUrl({
    group: "pitching",
    season,
    teamId,
    sportId,
    startDate: weekStart,
    endDate: weekEnd,
  });

  const [hitPayload, pitPayload] = await Promise.all([fetchJson(hitUrl), fetchJson(pitUrl)]);
  const hitSplits = getSplits(hitPayload);
  const pitSplits = getSplits(pitPayload);

  // Build maps: playerId -> stat object
  const hitById = new Map();
  for (const s of hitSplits) {
    const pid = num(s?.player?.id);
    if (!pid) continue;
    if (s?.stat && typeof s.stat === "object") hitById.set(pid, s.stat);
  }

  const pitById = new Map();
  for (const s of pitSplits) {
    const pid = num(s?.player?.id);
    if (!pid) continue;
    if (s?.stat && typeof s.stat === "object") pitById.set(pid, s.stat);
  }

  console.log("AAA_DEBUG_team_stats_counts", {
    team_id: teamId,
    team_name: teamName,
    parent_org: parentOrgName,
    season,
    week_start: weekStart,
    week_end: weekEnd,
    roster_count: roster.length,
    hit_splits: hitSplits.length,
    pit_splits: pitSplits.length,
    hit_mapped: hitById.size,
    pit_mapped: pitById.size,
    hit_url: hitUrl,
    pit_url: pitUrl,
  });

  let playersSeen = 0;
  let upserted = 0;
  let statErrors = 0;
  let upsertErrors = 0;

  for (const r of roster) {
    playersSeen += 1;

    const playerId = num(r?.person?.id);
    const playerName = safeStr(r?.person?.fullName) || safeStr(r?.person?.name) || null;

    const rawPos =
      safeStr(r?.position?.abbreviation) ||
      safeStr(r?.position?.code) ||
      safeStr(r?.position?.name) ||
      "UNK";

    if (!playerId || !playerName) {
      statErrors += 1;
      continue;
    }

    const hitStat = hitById.get(playerId) || null;
    const pitStat = pitById.get(playerId) || null;

    // If BOTH missing, it means no weekly split returned for this player
    if (!hitStat && !pitStat) {
      // still upsert identity-only row (keeps roster coverage), but will not produce pills
      statErrors += 1;
    }

    // Hitter fields
    const pa = num(hitStat?.plateAppearances);
    const bb = num(hitStat?.baseOnBalls);
    const so = num(hitStat?.strikeOuts);
    const hr = num(hitStat?.homeRuns);

    const avg = num(hitStat?.avg);
    const slg = num(hitStat?.slg);
    const iso = avg !== null && slg !== null ? Math.round((slg - avg) * 1000) / 1000 : null;

    // These are often absent in MLB API for MiLB; keep null unless present
    const ev90 = num(hitStat?.ev90) ?? null;
    const wrc_plus = num(hitStat?.wrcPlus) ?? null;

    // Pitcher fields
    const bf = num(pitStat?.battersFaced);
    const so_p = num(pitStat?.strikeOuts);
    const bb_allowed = num(pitStat?.baseOnBalls);

    const payload = {
      // identity / keys
      week_start: weekStart,
      level: "AAA",
      org_mlb_team: parentOrgName,
      player_id: playerId,
      player_name: playerName,
      position_group: String(rawPos).toUpperCase(),

      // hitters
      pa,
      bb,
      so,
      hr,
      iso,
      ev90,
      wrc_plus,

      // pitchers
      bf,
      so_p,
      bb_allowed,

      // provenance
      source: "mlb_api",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    const { error } = await sb.from("milb_raw_weekly").upsert(payload, {
      onConflict: "week_start,level,player_id",
    });

    if (error) {
      upsertErrors += 1;
      console.log("AAA_INGEST_upsert_error", {
        team_id: teamId,
        team_name: teamName,
        player_id: playerId,
        player_name: playerName,
        code: error.code,
        message: error.message,
        details: error.details,
        hint: error.hint,
      });
      continue;
    }

    upserted += 1;
  }

  console.log("AAA_INGEST_TEAM_DONE", {
    team: teamName,
    parent_org: parentOrgName,
    team_id: teamId,
    playersSeen,
    upserted,
    statErrors,
    upsertErrors,
    hitSplits: hitSplits.length,
    pitSplits: pitSplits.length,
  });

  return {
    teamId,
    teamName,
    parentOrgName,
    playersSeen,
    upserted,
    statErrors,
    upsertErrors,
    hitSplits: hitSplits.length,
    pitSplits: pitSplits.length,
  };
}

/* ---------------- ingest core entry ---------------- */

async function run(urlStr) {
  const url = new URL(urlStr);
  const mode = (url.searchParams.get("mode") || "run").trim();

  const teamIdRaw = url.searchParams.get("team_id");
  const teamId = teamIdRaw !== null ? Number(teamIdRaw) : null;

  const weekStart = (url.searchParams.get("week_start") || "").trim();
  const season = Number(url.searchParams.get("season"));

  if (mode === "status") {
    return {
      status: 200,
      body:
        "OK ingest-milb-aaa-weekly STATUS.\n" +
        "Single-team mode:\n" +
        "  ?team_id=...&week_start=YYYY-MM-DD&season=YYYY&token=ADMIN_RUN_TOKEN\n" +
        "Multi-org mode (omit team_id):\n" +
        "  ?week_start=YYYY-MM-DD&season=YYYY&max_teams=3&token=ADMIN_RUN_TOKEN\n" +
        "Env override:\n" +
        "  MILB_ORG_ALLOWLIST=Detroit Tigers,New York Yankees,...\n",
    };
  }

  if (!isAuthedFromUrl(urlStr)) return { status: 401, body: "Unauthorized" };

  if (!/^\d{4}-\d{2}-\d{2}$/.test(weekStart)) return { status: 400, body: "Missing/invalid week_start" };
  if (!Number.isFinite(season) || season < 2000) return { status: 400, body: "Missing/invalid season" };
  if (teamIdRaw !== null && (!Number.isFinite(teamId) || teamId <= 0))
    return { status: 400, body: "Missing/invalid team_id" };

  const weekEnd = addDays(weekStart, 6);
  const sportId = 11; // AAA fixed

  const { createClient } = await import("@supabase/supabase-js");
  const sb = createClient(mustEnv("SUPABASE_URL"), mustEnv("SUPABASE_SERVICE_ROLE_KEY"));

  // Single-team mode
  if (teamIdRaw !== null) {
    const result = await ingestOneTeam({ sb, teamId, weekStart, weekEnd, season, sportId });
    return {
      status: 200,
      body:
        `OK AAA ingest complete (single-team).\n` +
        `week_start=${weekStart}\n` +
        `week_end=${weekEnd}\n` +
        `season=${season}\n` +
        `team_id=${teamId}\n` +
        `team_name=${result.teamName}\n` +
        `parent_org=${result.parentOrgName || ""}\n` +
        `players_seen=${result.playersSeen}\n` +
        `upserted=${result.upserted}\n` +
        `stat_errors=${result.statErrors}\n` +
        `upsert_errors=${result.upsertErrors}\n` +
        `hit_splits=${result.hitSplits}\n` +
        `pit_splits=${result.pitSplits}\n`,
    };
  }

  // Multi-org mode
  const allowlist = getOrgAllowlist();
  const aaaTeams = await fetchAaaTeamsForSeason(season);

  const selectedAll = filterTeamsByOrgAllowlist(aaaTeams, allowlist);

  // max_teams limiter for timeout safety
  const maxTeamsParam = Number(url.searchParams.get("max_teams"));
  const maxTeams = Number.isFinite(maxTeamsParam) && maxTeamsParam > 0 ? maxTeamsParam : selectedAll.length;
  const selected = selectedAll.slice(0, maxTeams);

  console.log("AAA_MULTI_ORG_SELECTION", {
    season,
    week_start: weekStart,
    week_end: weekEnd,
    allowlist_count: allowlist.length,
    aaa_teams_total: aaaTeams.length,
    selected_total: selectedAll.length,
    selected_limited: selected.length,
    selected_preview: selected.slice(0, 20),
  });

  if (selected.length === 0) {
    return {
      status: 200,
      body:
        "OK AAA ingest (multi-org) found 0 matching AAA teams.\n" +
        "Check MILB_ORG_ALLOWLIST values and MLB Stats parentOrgName strings.\n",
    };
  }

  const summaries = [];
  let teamsOk = 0;
  let teamsFailed = 0;

  for (const t of selected) {
    try {
      const teamResult = await ingestOneTeam({
        sb,
        teamId: t.teamId,
        weekStart,
        weekEnd,
        season,
        sportId,
      });
      summaries.push({
        team_id: t.teamId,
        team_name: teamResult.teamName,
        parent_org: teamResult.parentOrgName,
        upserted: teamResult.upserted,
        players_seen: teamResult.playersSeen,
        stat_errors: teamResult.statErrors,
        upsert_errors: teamResult.upsertErrors,
      });
      teamsOk += 1;
    } catch (e) {
      teamsFailed += 1;
      console.log("AAA_INGEST_TEAM_FAILED", {
        team_id: t.teamId,
        team_name: t.name,
        parent_org: t.parentOrgName,
        error: e?.message || String(e),
      });
    }
  }

  const totalUpserted = summaries.reduce((a, s) => a + (Number(s.upserted) || 0), 0);
  const totalSeen = summaries.reduce((a, s) => a + (Number(s.players_seen) || 0), 0);

  return {
    status: 200,
    body:
      `OK AAA ingest complete (multi-org).\n` +
      `week_start=${weekStart}\n` +
      `week_end=${weekEnd}\n` +
      `season=${season}\n` +
      `org_allowlist=${allowlist.join(", ")}\n` +
      `teams_selected_total=${selectedAll.length}\n` +
      `teams_selected_limited=${selected.length}\n` +
      `teams_ok=${teamsOk}\n` +
      `teams_failed=${teamsFailed}\n` +
      `players_seen_total=${totalSeen}\n` +
      `upserted_total=${totalUpserted}\n` +
      `team_summaries=${JSON.stringify(summaries)}\n`,
  };
}

/* ---------------- v1 entrypoint ---------------- */

export const handler = async (event) => {
  try {
    const rawUrl = event?.rawUrl || "";
    const path = event?.path || "/.netlify/functions/ingest-milb-aaa-weekly";
    const full = rawUrl.startsWith("http")
      ? rawUrl
      : `https://diamondsignals.ai${path}${rawUrl.includes("?") ? rawUrl.slice(rawUrl.indexOf("?")) : ""}`;
    const out = await run(full);
    return {
      statusCode: out.status,
      headers: { "content-type": "text/plain; charset=utf-8" },
      body: out.body,
    };
  } catch (e) {
    return {
      statusCode: 500,
      headers: { "content-type": "text/plain; charset=utf-8" },
      body: `CRASH(v1): ${e?.message || e}\n${e?.stack || ""}`,
    };
  }
};

/* ---------------- v2 entrypoint ---------------- */

export default async (req) => {
  try {
    const out = await run(req.url);
    return new Response(out.body, { status: out.status });
  } catch (e) {
    return new Response(`CRASH(v2): ${e?.message || e}\n${e?.stack || ""}`, { status: 500 });
  }
};