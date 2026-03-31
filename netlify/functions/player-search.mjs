export default async (req) => {
  try {
    const url = new URL(req.url);
    const query = (url.searchParams.get("q") || "").trim();

    if (query.length < 2) {
      return json({
        query,
        results: []
      });
    }

    const requestUrl = new URL(req.url);
    const indexUrl = `${requestUrl.origin}/player_index.json`;

    const res = await fetch(indexUrl, {
  headers: {
    "accept": "application/json"
  }
});

const rawText = await res.text();
const contentType = res.headers.get("content-type") || "";

if (!res.ok) {
  return json(
    {
      query,
      results: [],
      error: "player_index_unavailable",
      debug: {
        indexUrl,
        status: res.status,
        contentType,
        preview: rawText.slice(0, 200)
      }
    },
    500
  );
}

let payload;

try {
  payload = JSON.parse(rawText);
} catch (error) {
  return json(
    {
      query,
      results: [],
      error: String(error),
      debug: {
        indexUrl,
        status: res.status,
        contentType,
        preview: rawText.slice(0, 200)
      }
    },
    500
  );
}

    const payload = await res.json();
    const players = Array.isArray(payload?.players) ? payload.players : [];

    const normalizedQuery = normalizeText(query);

    const scored = players
      .map((player) => ({
        ...player,
        _score: scorePlayerMatch(normalizedQuery, player)
      }))
      .filter((player) => player._score !== null)
      .sort((a, b) => compareScores(a._score, b._score))
      .slice(0, 6)
      .map(({ _score, ...player }) => ({
        player_id: player.player_id,
        full_name: player.full_name,
        team: player.team || "",
        team_name: player.team_name || "",
        position: player.position || "",
        bats: player.bats || "",
        throws: player.throws || "",
        status: player.status || "",
        headshot_url: player.headshot_url || buildHeadshotUrl(player.player_id),
        profile_url: `/scout/${player.player_id}`
      }));

    return json({
      query,
      results: scored
    });
  } catch (error) {
    return json(
      {
        query: "",
        results: [],
        error: String(error)
      },
      500
    );
  }
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  });
}

function normalizeText(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\./g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function buildHeadshotUrl(playerId) {
  return `https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_100/v1/people/${playerId}/headshot/67/current`;
}

function scorePlayerMatch(query, player) {
  const fullName = normalizeText(player.full_name || "");
  const firstName = normalizeText(player.first_name || "");
  const lastName = normalizeText(player.last_name || "");
  const team = normalizeText(player.team || "");
  const status = normalizeText(player.status || "");

  if (!fullName && !lastName) {
    return null;
  }

  let tier = null;

  if (fullName === query) {
    tier = 0;
  } else if (lastName === query) {
    tier = 1;
  } else if (fullName.startsWith(query)) {
    tier = 2;
  } else if (lastName.startsWith(query)) {
    tier = 3;
  } else if (`${firstName} ${lastName}`.startsWith(query)) {
    tier = 4;
  } else if (fullName.includes(query)) {
    tier = 5;
  } else if (lastName.includes(query)) {
    tier = 6;
  } else if (team && team === query) {
    tier = 7;
  } else {
    return null;
  }

  const activeBoost = status === "active" ? 0 : 1;
  const nameLengthPenalty = Math.abs(fullName.length - query.length);

  return [tier, activeBoost, nameLengthPenalty, fullName];
}

function compareScores(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}