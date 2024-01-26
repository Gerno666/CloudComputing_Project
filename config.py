from os import environ as env
import multiprocessing

PORT = int(env.get("PORT", 8001))
DEBUG_MODE = int(env.get("DEBUG_MODE", 1))

# Local
DATASET = 'survey_results_public.csv'

# Model Name
MODEL = 'tensorflow_model'

# Gunicorn config
bind = ":" + str(PORT)
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2 * multiprocessing.cpu_count()