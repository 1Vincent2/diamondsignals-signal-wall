create table if not exists prospect_intelligence_daily (
  snapshot_date date not null,
  player_id bigint not null,
  player_name text not null,
  org text,
  level text,
  signal_type text,
  position_group text,

  edge_score numeric,
  score_version text,
  source_badge text,

  data_freshness_hours numeric,
  latency_warning boolean default false,

  signal_archetype text,

  is_recent_arrival boolean default false,
  arrival_type text,
  arrival_date timestamptz,

  last_7d_iso numeric,
  k_bb_ratio numeric,
  exit_velo_90th numeric,

  bb_rate numeric,
  k_rate numeric,
  k_rate_proxy numeric,
  bb_rate_proxy numeric,

  bf integer,
  pa integer,

  trend_points text,
  trend_glow boolean default false,

  scout_narrative text,
  created_at timestamptz not null default now(),

  constraint prospect_intelligence_daily_snapshot_player_key
    unique (snapshot_date, player_id)
);

create index if not exists prospect_intelligence_daily_snapshot_date_idx
  on prospect_intelligence_daily (snapshot_date desc);

create index if not exists prospect_intelligence_daily_edge_score_idx
  on prospect_intelligence_daily (edge_score desc);

create index if not exists prospect_intelligence_daily_level_idx
  on prospect_intelligence_daily (level);

create index if not exists prospect_intelligence_daily_recent_arrival_idx
  on prospect_intelligence_daily (is_recent_arrival);