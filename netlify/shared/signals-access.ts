const TOKEN_VERSION = "v1";
const MAX_AGE_SECONDS = 60 * 60 * 24 * 365;

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });

  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function fromBase64Url(value: string): Uint8Array {
  const normalized = value
    .replace(/-/g, "+")
    .replace(/_/g, "/");

  const padded =
    normalized + "=".repeat((4 - (normalized.length % 4)) % 4);

  const binary = atob(padded);

  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

async function importSigningKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}

export async function createSignalsAccessToken(
  email: string,
  secret: string
): Promise<string> {
  const normalizedEmail = email.trim().toLowerCase();

  const payload = {
    v: TOKEN_VERSION,
    email: normalizedEmail,
    issuedAt: Math.floor(Date.now() / 1000),
  };

  const encodedPayload = toBase64Url(
    new TextEncoder().encode(JSON.stringify(payload))
  );

  const key = await importSigningKey(secret);

  const signature = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(encodedPayload)
  );

  return `${encodedPayload}.${toBase64Url(new Uint8Array(signature))}`;
}

export async function verifySignalsAccessToken(
  token: string,
  secret: string
): Promise<boolean> {
  try {
    if (!token || !secret) return false;

    const parts = token.split(".");
    if (parts.length !== 2) return false;

    const [encodedPayload, encodedSignature] = parts;

    const key = await importSigningKey(secret);

    const validSignature = await crypto.subtle.verify(
      "HMAC",
      key,
      fromBase64Url(encodedSignature),
      new TextEncoder().encode(encodedPayload)
    );

    if (!validSignature) return false;

    const payloadText = new TextDecoder().decode(
      fromBase64Url(encodedPayload)
    );

    const payload = JSON.parse(payloadText);

    if (
      payload?.v !== TOKEN_VERSION ||
      typeof payload?.email !== "string" ||
      typeof payload?.issuedAt !== "number"
    ) {
      return false;
    }

    const age = Math.floor(Date.now() / 1000) - payload.issuedAt;

    return age >= 0 && age <= MAX_AGE_SECONDS;
  } catch {
    return false;
  }
}

export const SIGNALS_ACCESS_COOKIE = "ds_signals_access";
export const SIGNALS_ACCESS_MAX_AGE = MAX_AGE_SECONDS;
