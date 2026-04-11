# Canonical Dossier Data Contract v1

## Purpose
This contract defines the single authoritative player truth layer for DiamondSignals.
All signal pages may add page-specific context, but core player identity and canonical
core metrics must come from one shared dossier-ready record.

## Universal Player Truth
These fields must be consistent across all pages and dossier views.

- player_id
- player_name
- team
- position
- age
- bats
- throws
- headshot_url
- canonical_score
- canonical_score_label
- trend_points_7d
- trend_note
- rostered_by_user

## Page-Context-Specific Insights
These fields may differ by page and should not override canonical truth.

### Signal Wall context
- signal_type
- signal_reason
- signal_date
- page_specific_badges

### Call-Up context
- promotion_window
- movement_context
- aaa_signal_context
- page_specific_badges

### Hidden Gems context
- divergence_context
- under_the_hood_traits
- market_context
- page_specific_badges

### Lab context
- raw percentile context
- experimental metrics
- page_specific_badges

## Rules
1. player_id is the anchor identity field.
2. canonical_score must come from one shared dossier-ready pipeline output.
3. page-specific pages may explain why a player is on that page, but may not redefine the canonical player identity.
4. dossier pages must use the canonical truth layer first, then layer in deeper contextual detail.
5. any player action launcher must resolve through the same canonical player_id.

## Short-Term Implementation
Use a generated JSON middleware layer:
- dossier_canon.json

This JSON becomes the canonical source for:
- player identity
- canonical score
- trend
- core dossier truth

## Long-Term Implementation
After validating the contract in production, migrate the proven canonical fields
into a native Supabase table if needed.