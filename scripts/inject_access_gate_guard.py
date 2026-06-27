from pathlib import Path

DIST_DIR = Path("dist")

GUARD = r'''<script>
(function () {
  var protectedHost = "signals.diamondsignals.ai";
  var gatePath = "/";
  var path = window.location.pathname || "/";

  var isGatePage = path === "/" || path === "/index.html";

  /*
    PUBLIC_PROMO_PATHS
    Temporary marketing allowlist for X / Reddit traffic.
    These pages remain public while the rest of the Signal Wall system stays gated.
  */
  var publicPromoPaths = [
    "/live/",
    "/admin/kinetic-drift/",
    "/waiver-wire/"
  ];

  var isPublicPromoPath = publicPromoPaths.some(function(publicPath) {
    return path === publicPath || path.indexOf(publicPath) === 0;
  });

  var isLocal =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "";

  /*
    ACCESS_GATE_PREVIEW_HOST_ALLOWLIST_V1
    Allows the quarantined Netlify preview project to serve branch pages
    without escaping back to production. Production host gating remains intact.
  */
  var isPreviewHost =
    window.location.hostname === "diamondsignals-mobile-preview.netlify.app";

  if (isLocal || isPreviewHost || isGatePage || isPublicPromoPath) return;

  function readCookie(name) {
    return document.cookie
      .split(";")
      .map(function (part) { return part.trim(); })
      .filter(function (part) { return part.indexOf(name + "=") === 0; })
      .map(function (part) { return decodeURIComponent(part.slice(name.length + 1)); })[0] || "";
  }

  var hasAccess =
    readCookie("ds_founding_access") === "1" ||
    readCookie("ds_captured") === "1";

  if (!hasAccess) {
    var next = encodeURIComponent(path + (window.location.search || ""));
    window.location.replace("https://" + protectedHost + gatePath + "?next=" + next);
  }
})();
</script>'''

SKIP_RELATIVE = {
    "index.html",  # front-door gate
}

def should_skip(path: Path) -> bool:
    rel = path.relative_to(DIST_DIR).as_posix()
    if rel in SKIP_RELATIVE:
        return True

    # Public JSON/API-ish and reports without a normal head are naturally ignored later.
    return False

def main() -> None:
    if not DIST_DIR.exists():
        raise SystemExit("dist directory not found. Run the dashboard build first.")

    patched = []
    skipped = []

    for path in sorted(DIST_DIR.rglob("*.html")):
        rel = path.relative_to(DIST_DIR).as_posix()

        if should_skip(path):
            skipped.append(rel)
            continue

        text = path.read_text(encoding="utf-8")

        if "ACCESS_GATE_GUARD_V1" in text:
            skipped.append(rel)
            continue

        if "</head>" not in text:
            skipped.append(rel)
            continue

        insert = "\n<!-- ACCESS_GATE_GUARD_V1 -->\n" + GUARD + "\n"
        text = text.replace("</head>", insert + "</head>", 1)
        path.write_text(text, encoding="utf-8")
        patched.append(rel)

    print("ACCESS_GATE_GUARD_PATCHED")
    for item in patched:
        print(f"PATCHED {item}")

    print("ACCESS_GATE_GUARD_SKIPPED")
    for item in skipped:
        print(f"SKIPPED {item}")

if __name__ == "__main__":
    main()
