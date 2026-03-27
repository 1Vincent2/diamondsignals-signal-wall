export default async (request: Request, context: any) => {
  const url = new URL(request.url)
  const host = url.hostname
  const path = url.pathname

  if (host !== "signals.diamondsignals.ai") {
    return context.next()
  }

  const publicPrefixes = [
    "/favicon",
    "/robots.txt",
    "/signals.json",
  ]

  const isPublic = publicPrefixes.some((prefix) => path.startsWith(prefix))
  if (isPublic) {
    return context.next()
  }

  const cookies = request.headers.get("cookie") || ""
  const hasAccess =
    cookies.includes("ds_founding_access=1") ||
    cookies.includes("__Secure-ds_founding_access=1")

  if (hasAccess) {
    return context.next()
  }

  const redirectTo = new URL("https://diamondsignals.ai/founding-access/")
  redirectTo.searchParams.set("source", "signals")
  redirectTo.searchParams.set("next", `${url.pathname}${url.search}`)

  return Response.redirect(redirectTo.toString(), 302)
}

export const config = {
  path: "/*",
}