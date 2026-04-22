const { createClient } = require("@supabase/supabase-js");

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

    const sb = createClient(supabaseUrl, supabaseKey, {
      auth: { persistSession: false },
    });

    const payload = {
      email,
      source,
      entry_surface,
      referrer: referrer || null,
      utm_source: utm_source || null,
      utm_medium: utm_medium || null,
      utm_campaign: utm_campaign || null,
      user_agent: user_agent || null,
      capture_status: "captured",
      last_seen_at: new Date().toISOString(),
    };

    const { data, error } = await sb
      .from("founding_access")
      .upsert(payload, { onConflict: "email" })
      .select("email")
      .limit(1);

    if (error) {
      return json(500, {
        ok: false,
        error: "Supabase upsert failed",
        details: error.message,
      });
    }

    return json(200, {
      ok: true,
      captured: true,
      email,
      next_url: "/live/",
      record: data?.[0] || null,
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
      "Cache-Control": "no-store"
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
