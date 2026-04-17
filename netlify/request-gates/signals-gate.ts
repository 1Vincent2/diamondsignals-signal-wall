export default async (_request: Request, context: any) => {
  return context.next()
}

export const config = {
  path: "/*",
}
