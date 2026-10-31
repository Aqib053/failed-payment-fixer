from supabase import Client, create_client

from app import config

db: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
