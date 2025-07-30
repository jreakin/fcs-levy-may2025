from django.urls import path
from . import views

urlpatterns = [
    # Web interface URLs
    path('', views.home, name='home'),
    path('upload/', views.upload_voter_file, name='upload_voter_file'),
    path('voters/', views.voter_list, name='voter_list'),
    path('voters/<int:voter_id>/', views.voter_detail, name='voter_detail'),
    path('voters/<int:voter_id>/verify/', views.verify_voter_address, name='verify_voter_address'),
    path('elections/', views.election_list, name='election_list'),
    path('ajax/office-types/', views.get_office_types, name='get_office_types'),
    
    # API endpoints
    path('api/elections/', views.api_elections, name='api_elections'),
    path('api/voters/<int:voter_id>/election-data/', views.api_voter_election_data, name='api_voter_election_data'),
    path('api/voters/<int:voter_id>/address/', views.api_voter_address, name='api_voter_address'),
    path('api/voters/<int:voter_id>/verify-address/', views.api_verify_address, name='api_verify_address'),
    path('api/voters/<int:voter_id>/district/', views.api_voter_district, name='api_voter_district'),
]