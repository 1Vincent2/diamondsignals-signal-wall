import {
  SIGNALS_ACCESS_COOKIE,
  verifySignalsAccessToken,
} from "../shared/signals-access.ts";

function readCookie(request: Request, name: string): string {
  const cookieHeader = request.headers.get("cookie") || "";

  for (const part of cookieHeader.split(";")) {
    const trimmed = part.trim();
    const prefix = `${name}=`;

    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }

  return "";
}

function isPublicPath(pathname: string): boolean {
  /*
    MOBILE COMMAND PREVIEW ONLY.
    Path-scoped canary exception; remove before production merge.
  */
  if (
    pathname === "/mobile-live-canary" ||
    pathname === "/mobile-live-canary/" ||
    pathname.startsWith("/mobile-live-canary/") ||
    pathname === "/mobile-promotion-watch-canary" ||
    pathname === "/mobile-promotion-watch-canary/" ||
    pathname.startsWith("/mobile-promotion-watch-canary/") ||
    pathname === "/mobile-velocity-decay-canary" ||
    pathname === "/mobile-velocity-decay-canary/" ||
    pathname.startsWith("/mobile-velocity-decay-canary/") ||
    pathname === "/mobile-stuff-disruption-canary" ||
    pathname === "/mobile-stuff-disruption-canary/" ||
    pathname.startsWith("/mobile-stuff-disruption-canary/") ||
    pathname === "/mobile-ivb-heat-map-canary" ||
    pathname === "/mobile-ivb-heat-map-canary/" ||
    pathname.startsWith("/mobile-ivb-heat-map-canary/") ||
    pathname === "/mobile-apex-extraction-canary" ||
    pathname === "/mobile-apex-extraction-canary/" ||
    pathname.startsWith("/mobile-apex-extraction-canary/") ||
    pathname === "/mobile-mlb-extraction-canary" ||
    pathname === "/mobile-mlb-extraction-canary/" ||
    pathname.startsWith("/mobile-mlb-extraction-canary/")
  ) {
    return true;
  }

  if (pathname === "/" || pathname === "/index.html") {
    return true;
  }

  if (pathname.startsWith("/.netlify/")) {
    return true;
  }

  /*
    Static assets must remain public so the front door and protected
    pages can load their presentation resources normally.
  */
  if (
    pathname.startsWith("/assets/") ||
    pathname.startsWith("/css/") ||
    pathname.startsWith("/js/") ||
    pathname.startsWith("/images/") ||
    pathname.startsWith("/fonts/")
  ) {
    return true;
  }

  /*
    Machine-readable resources used by the authenticated app and
    operational/status tooling are not browser intelligence surfaces.
  */
  if (
    pathname === "/dossier_canon.json" ||
    pathname === "/scout_metrics.json" ||
    pathname === "/player_index.json" ||
    pathname === "/admin/player_signal_index.json" ||
    pathname.startsWith("/status/")
  ) {
    return true;
  }

  /*
    Ordinary static files such as icons, manifests, stylesheets,
    scripts, images and fonts are not Signal Wall content pages.
  */
  if (
    /\.(?:css|js|mjs|map|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|otf|txt|xml|webmanifest)$/i.test(
      pathname
    )
  ) {
    return true;
  }

  return false;
}

function safeReturnDestination(url: URL): string {
  const destination = url.pathname + url.search;

  if (
    destination.startsWith("/") &&
    !destination.startsWith("//")
  ) {
    return destination;
  }

  return "/live/";
}

export default async (request: Request, context: any) => {
  const url = new URL(request.url);

  if (
    url.pathname === "/hidden-gems" ||
    url.pathname === "/hidden-gems/"
  ) {
    url.pathname = "/mlb-extraction/";
    return Response.redirect(url.toString(), 301);
  }

  /*
    The root remains the public front door for new visitors.
    Returning visitors with a valid signed access credential
    should skip the Unlock Access screen and go directly to
    the live Signal Wall.
  */
  if (url.pathname === "/" || url.pathname === "/index.html") {
    const rootSecret = String(
      Netlify.env.get("SIGNALS_ACCESS_SECRET") || ""
    ).trim();

    if (rootSecret) {
      const rootToken = readCookie(
        request,
        SIGNALS_ACCESS_COOKIE
      );

      const rootHasAccess = await verifySignalsAccessToken(
        rootToken,
        rootSecret
      );

      if (rootHasAccess) {
        const liveUrl = new URL("/live/", url.origin);
        return Response.redirect(liveUrl.toString(), 302);
      }
    }

    return context.next();
  }

  if (isPublicPath(url.pathname)) {
    return context.next();
  }

  const secret = String(
    Netlify.env.get("SIGNALS_ACCESS_SECRET") || ""
  ).trim();

  /*
    Fail closed on protected content if production configuration
    is incomplete. The front door remains public so access can
    never turn into a redirect loop.
  */
  if (!secret) {
    return new Response(
      "Signal Wall access configuration unavailable.",
      {
        status: 503,
        headers: {
          "Cache-Control": "no-store",
          "Content-Type": "text/plain; charset=utf-8",
        },
      }
    );
  }

  const token = readCookie(
    request,
    SIGNALS_ACCESS_COOKIE
  );

  const hasAccess = await verifySignalsAccessToken(
    token,
    secret
  );

  if (hasAccess) {
    return context.next();
  }

  const gateUrl = new URL("/", url.origin);

  gateUrl.searchParams.set(
    "next",
    safeReturnDestination(url)
  );

  return Response.redirect(gateUrl.toString(), 302);
};

export const config = {
  path: "/*",
};
