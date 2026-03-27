export default async (request: Request, context: any) => {
  const url = new URL(request.url)
  const host = url.hostname

  if (host !== "signals.diamondsignals.ai") {
    return context.next()
  }

  const redirectTo = new URL("https://diamondsignals.ai/founding-access/")
  redirectTo.searchParams.set("source", "signals")
  redirectTo.searchParams.set("next", `${url.pathname}${url.search}`)
  redirectTo.searchParams.set("gate_test", "1")

  return Response.redirect(redirectTo.toString(), 302)
}

export const config = {
  path: "/*",
}