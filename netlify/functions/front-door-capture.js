exports.handler = async (event) => {
  try {
    if (event.httpMethod === "OPTIONS") {
      return json(204, "");
    }

    if (event.httpMethod !== "POST") {
      return json(405, { ok: false, error: "Method not allowed" });
    }

    const body = safeJson(event.body || "{}");
    const email = cleanEmail(body.email || "");
    const source = String(body.source || "signals_front_door").trim();
    const entry_surface = String(body.entry_surface || "signals_subdomain").trim();
    const referrer = String(body.referrer || "").trim();
    const utm_source = String(body.utm_source || "").trim();
    const utm_medium = String(body.utm_medium || "").trim();
    const utm_campaign = String(body.utm_campaign || "").trim();
    const first_name = String(body.first_name || "").trim();
    const user_agent = String(
      event.headers["user-agent"] ||
      event.headers["User-Agent"] ||
      ""
    ).trim();

    if (!email || !isValidEmail(email)) {
      return json(400, { ok: false, error: "Invalid email address" });
    }

    const supabaseUrl = String(process.env.SUPABASE_URL || "").trim();
    const supabaseKey = String(process.env.SUPABASE_SERVICE_ROLE_KEY || "").trim();

    if (!supabaseUrl || !supabaseKey) {
      return json(500, { ok: false, error: "Missing Supabase environment variables" });
    }

    const payload = {
      email,
      source,
      entry_surface,
      first_name: first_name || null,
      referrer: referrer || null,
      utm_source: utm_source || null,
      utm_medium: utm_medium || null,
      utm_campaign: utm_campaign || null,
      user_agent: user_agent || null,
      capture_status: "captured",
      last_seen_at: new Date().toISOString(),
    };

    const endpoint =
      supabaseUrl.replace(/\/+$/, "") +
      "/rest/v1/founding_access?on_conflict=email";

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: {
        apikey: supabaseKey,
        Authorization: "Bearer " + supabaseKey,
        "Content-Type": "application/json",
        Prefer: "resolution=merge-duplicates,return=representation",
      },
      body: JSON.stringify(payload),
    });

    const data = await resp.json().catch(() => ({}));

    if (!resp.ok) {
      return json(500, {
        ok: false,
        error: "Supabase upsert failed",
        details: data,
      });
    }

    let welcome = { attempted: false, ok: false };

    try {
      const welcomeResp = await fetch(
        "https://diamondsignals.ai/.netlify/functions/system-send-welcome",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email,
            name: first_name || "",
          }),
        }
      );

      const welcomeData = await welcomeResp.json().catch(() => ({}));

      welcome = {
        attempted: true,
        ok: Boolean(welcomeResp.ok && welcomeData && welcomeData.ok),
        status: welcomeResp.status,
        response: welcomeData,
      };
    } catch (err) {
      welcome = {
        attempted: true,
        ok: false,
        error: String(err?.message || err),
      };
    }

    return json(200, {
      ok: true,
      captured: true,
      email,
      next_url: "/live/",
      record: Array.isArray(data) ? data[0] || null : data,
      welcome,
    });
  } catch (err) {
    return json(500, {
      ok: false,
      error: String(err?.message || err),
    });
  }
};

function json(statusCode, body) {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
    body: typeof body === "string" ? body : JSON.stringify(body),
  };
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    return {};
  }
}

function cleanEmail(input) {
  return String(input || "")
    .trim()
    .toLowerCase()
    .replace(/^<|>$/g, "")
    .replaceAll(" ", "");
}

function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}
