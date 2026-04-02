# WORKFLOW

## Active source of truth
- `dashboard/`
- `dashboard/templates/`

## Generated output
- `dist/`

## Inactive / archived staging work
- `archive/src-staging/`

## Normal workflow
1. Edit files in `dashboard/` or `dashboard/templates/`
2. Rebuild locally
3. Inspect generated output in `dist/`
4. Commit source files and generated output when needed for deploy consistency

## Notes
- Do not treat `dist/` as primary source
- Do not edit archived staging files unless intentionally reviving that path