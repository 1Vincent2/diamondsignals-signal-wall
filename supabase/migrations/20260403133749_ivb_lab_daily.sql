create table if not exists ivb_lab_daily (
  report_date date not null,
  player_id bigint not null,
  player_name text not null,
  team text,
  pitch_count integer,
  avg_fastball_velo numeric,
  ivb_raw numeric,
  ivb_vs_avg numeric,
  dead_zone_flag boolean default false,
  contact_risk_flag boolean default false,
  whiff_probability text,
  climber_delta numeric,
  climber_flag boolean default false,
  heat_band text,
  vaa numeric,
  created_at timestamptz not null default now(),

  constraint ivb_lab_daily_report_player_key unique (report_date, player_id)
);

create index if not exists ivb_lab_daily_report_date_idx
  on ivb_lab_daily (report_date desc);

create index if not exists ivb_lab_daily_heat_band_idx
  on ivb_lab_daily (heat_band);

create index if not exists ivb_lab_daily_climber_flag_idx
  on ivb_lab_daily (climber_flag);

create index if not exists ivb_lab_daily_dead_zone_flag_idx
  on ivb_lab_daily (dead_zone_flag);
  