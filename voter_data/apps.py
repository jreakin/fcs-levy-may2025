from django.apps import AppConfig


class VoterDataConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'voter_data'
    verbose_name = 'Voter Data Management'