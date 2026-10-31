-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.

create table if not exists transactions (
  id          text primary key,
  customer    text not null,
  amount      numeric not null,
  payee       text not null,
  payee_vpa   text,
  bank        text,
  -- success | pending (debited, not credited) | refund_pending (failed, refund not back)
  -- | reversed | repushed
  status      text not null,
  stage       text,
  created_at  timestamptz not null default now()
);

create table if not exists tickets (
  id          bigint generated always as identity primary key,
  txn_id      text references transactions(id) on delete cascade,
  kind        text not null,          -- dispute | escalation | compensation
  summary     text,
  amount_inr  numeric default 0,
  created_by  text default 'resolver', -- resolver | shield | human
  created_at  timestamptz not null default now()
);

-- Lock tables to the backend only (the service role key skips RLS).
alter table transactions enable row level security;
alter table tickets enable row level security;
