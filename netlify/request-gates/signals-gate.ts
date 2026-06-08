export default async (request: Request, context: any) => {
  const url = new URL(request.url)

  if (url.pathname === "/hidden-gems" || url.pathname === "/hidden-gems/") {
    url.pathname = "/mlb-extraction/"
    return Response.redirect(url.toString(), 301)
  }

  return context.next()
}

export const config = {
  path: "/*",
}
