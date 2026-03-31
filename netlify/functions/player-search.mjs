export async function handler(event) {
  try {
    const q = String(event.queryStringParameters?.q || "")
      .trim()
      .toLowerCase();

    if (!q || q.length < 2) {
      return jsonResponse(200, { results: [] });
    }

    const baseUrl =
      process.env.URL ||
      process.env.DEPLOY_URL ||
      "https://signals.diamondsignals.ai";

    const indexUrl = `${baseUrl.replace(/\/$/, "")}/player_index.json`;
    const res = await fetch(indexUrl);

    if (!res.ok) {
      return jsonResponse(500, {
        error: "player_index_fetch_failed",
        detail: `Could not load player_index.json (${res.status})`,
      });
    }

    const payload = await res.json();
    const players = Array.isArray(payload?.players) ? payload.players : [];

    const results = players
      .filter((player) => matchesQuery(player, q))
      .slice(0, 8)
      .map((player) => ({
        player_id: player.player_id,
        full_name: player.full_name || "Unknown Player",
        team: player.team || "",
        position: player.position || "",
        headshot_url: player.headshot_url || "",
        profile_url: `/scout/${player.player_id}/`,
      }));

    return jsonResponse(200, { results });
  } catch (error) {
    return jsonResponse(500, {
      error: "search_failed",
      detail: String(error?.message || error),
    });
  }
}

function matchesQuery(player, q) {
  const fullName = String(player?.full_name || "").toLowerCase();
  const firstName = String(player?.first_name || "").toLowerCase();
  const lastName = String(player?.last_name || "").toLowerCase();
  const team = String(player?.team || "").toLowerCase();
  const teamName = String(player?.team_name || "").toLowerCase();
  const position = String(player?.position || "").toLowerCase();
  const playerId = String(player?.player_id || "").toLowerCase();

  return (
    fullName.includes(q) ||
    firstName.includes(q) ||
    lastName.includes(q) ||
    team.includes(q) ||
    teamName.includes(q) ||
    position.includes(q) ||
    playerId.includes(q)
  );
}

function jsonResponse(statusCode, payload) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "access-control-allow-origin": "*",
    },
    body: JSON.stringify(payload),
  };
}