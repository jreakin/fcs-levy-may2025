from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import transaction, models
import csv
import io
import json

try:
    from rest_framework.decorators import api_view, permission_classes
    from rest_framework.permissions import IsAuthenticated
    from rest_framework.response import Response
    from rest_framework import status
    DRF_AVAILABLE = True
except ImportError:
    DRF_AVAILABLE = False
    # Mock decorators when DRF is not available
    def api_view(methods):
        def decorator(func):
            return func
        return decorator
    
    def permission_classes(perms):
        def decorator(func):
            return func
        return decorator
    
    class Response:
        def __init__(self, data, status=200):
            self.data = data
            self.status_code = status

from .models import VoterRecord, Election, ElectionData
from .utils import (
    construct_address_lines, 
    update_addresses, 
    get_office_types_for_district_level,
    validate_district_and_office,
    parse_csv_header_for_elections,
    clean_and_validate_voter_data
)
from .tasks import verify_address, batch_verify_addresses, get_address_verification_status


def home(request):
    """Home page view."""
    return render(request, 'voter_data/home.html')


def upload_voter_file(request):
    """
    Handle CSV file upload with election metadata and voter data processing.
    """
    if request.method == 'GET':
        # If there's a CSV file in the session, show the election metadata form
        if 'csv_data' in request.session:
            csv_data = request.session['csv_data']
            header = csv_data['header']
            return render(request, 'voter_data/upload_election_metadata.html', {
                'header': header,
                'district_levels': VoterRecord.DISTRICT_LEVEL_CHOICES,
            })
        
        # Otherwise show the file upload form
        return render(request, 'voter_data/upload_voter_file.html')
    
    elif request.method == 'POST':
        # Handle file upload
        if 'file' in request.FILES:
            return handle_file_upload(request)
        
        # Handle election metadata submission
        elif 'csv_data' in request.session:
            return handle_election_metadata(request)
        
        else:
            messages.error(request, 'No file uploaded or session expired.')
            return redirect('upload_voter_file')


def handle_file_upload(request):
    """Handle the initial CSV file upload."""
    file = request.FILES['file']
    
    if not file.name.endswith('.csv'):
        messages.error(request, 'Please upload a CSV file.')
        return redirect('upload_voter_file')
    
    try:
        # Read and parse CSV
        decoded_file = file.read().decode('utf-8')
        csv_reader = csv.reader(io.StringIO(decoded_file))
        header = next(csv_reader)
        rows = list(csv_reader)
        
        # Store CSV data in session
        request.session['csv_data'] = {
            'header': header,
            'rows': rows,
            'filename': file.name
        }
        
        # Redirect to election metadata form
        return redirect('upload_voter_file')
        
    except Exception as e:
        messages.error(request, f'Error reading CSV file: {str(e)}')
        return redirect('upload_voter_file')


def handle_election_metadata(request):
    """Handle election metadata submission and process the CSV data."""
    csv_data = request.session.get('csv_data')
    if not csv_data:
        messages.error(request, 'Session expired. Please upload the file again.')
        return redirect('upload_voter_file')
    
    try:
        with transaction.atomic():
            # Process election metadata
            elections = []
            header = csv_data['header']
            
            for col in header:
                election_name = request.POST.get(f'election_{col}')
                if election_name:
                    election_data = {
                        'name': election_name,
                        'year': request.POST.get(f'year_{col}'),
                        'election_type': request.POST.get(f'type_{col}'),
                        'date': request.POST.get(f'date_{col}'),
                        'column': col,
                        'data_type': request.POST.get(f'data_type_{col}')
                    }
                    elections.append(election_data)
            
            # Create or get Election instances
            election_instances = []
            for election in elections:
                elec, created = Election.objects.get_or_create(
                    name=election['name'],
                    defaults={
                        'election_type': election['election_type'],
                        'year': int(election['year']) if election['year'] else None,
                        'date': election['date'] if election['date'] else None
                    }
                )
                election_instances.append((elec, election['column'], election['data_type']))
            
            # Get district information
            district_level = request.POST.get('district_level', '')
            office_type = request.POST.get('office_type', '')
            
            # Validate district and office type combination
            if district_level and office_type:
                if not validate_district_and_office(district_level, office_type):
                    messages.error(request, f'Invalid office type "{office_type}" for district level "{district_level}"')
                    return redirect('upload_voter_file')
            
            # Process voter data
            processed_count = 0
            error_count = 0
            voter_ids_for_verification = []
            
            for row in csv_data['rows']:
                try:
                    data = dict(zip(header, row))
                    cleaned_data = clean_and_validate_voter_data(data)
                    
                    if not cleaned_data.get('vuid'):
                        error_count += 1
                        continue
                    
                    # Create or update voter record
                    voter, created = VoterRecord.objects.update_or_create(
                        vuid=cleaned_data['vuid'],
                        defaults={
                            **cleaned_data,
                            'district_level': district_level,
                            'office_type': office_type,
                            'file_origin': csv_data['filename']
                        }
                    )
                    
                    # Update addresses
                    update_addresses(voter)
                    voter_ids_for_verification.append(voter.id)
                    
                    # Link election data
                    for election, col, data_type in election_instances:
                        if data.get(col):
                            ElectionData.objects.update_or_create(
                                voter=voter,
                                election=election,
                                data_type=data_type,
                                defaults={'value': data[col]}
                            )
                    
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing row: {str(e)}")
            
            # Start batch address verification
            if voter_ids_for_verification:
                batch_verify_addresses.delay(voter_ids_for_verification)
            
            # Clear session data
            if 'csv_data' in request.session:
                del request.session['csv_data']
            
            messages.success(request, f'Successfully processed {processed_count} voter records. {error_count} errors occurred.')
            return redirect('voter_list')
            
    except Exception as e:
        messages.error(request, f'Error processing data: {str(e)}')
        return redirect('upload_voter_file')


def voter_list(request):
    """Display paginated list of voter records."""
    voters = VoterRecord.objects.all().order_by('lname', 'fname')
    
    # Search functionality
    search = request.GET.get('search')
    if search:
        voters = voters.filter(
            models.Q(fname__icontains=search) |
            models.Q(lname__icontains=search) |
            models.Q(vuid__icontains=search)
        )
    
    # Pagination
    paginator = Paginator(voters, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'voter_data/voter_list.html', {
        'page_obj': page_obj,
        'search': search
    })


def voter_detail(request, voter_id):
    """Display detailed view of a voter record."""
    voter = get_object_or_404(VoterRecord, id=voter_id)
    verification_status = get_address_verification_status(voter)
    election_data = voter.election_data.all().select_related('election')
    
    return render(request, 'voter_data/voter_detail.html', {
        'voter': voter,
        'verification_status': verification_status,
        'election_data': election_data
    })


@require_http_methods(["POST"])
def verify_voter_address(request, voter_id):
    """Trigger address verification for a specific voter."""
    voter = get_object_or_404(VoterRecord, id=voter_id)
    
    # Start verification task
    verify_address.delay(voter.id)
    
    messages.success(request, f'Address verification started for {voter.get_full_name()}')
    return redirect('voter_detail', voter_id=voter.id)


def election_list(request):
    """Display list of elections."""
    elections = Election.objects.all()
    return render(request, 'voter_data/election_list.html', {
        'elections': elections
    })


def get_office_types(request):
    """AJAX endpoint to get office types for a district level."""
    district_level = request.GET.get('district_level')
    office_types = get_office_types_for_district_level(district_level)
    
    return JsonResponse({
        'office_types': [{'value': office.lower().replace(' ', '_'), 'label': office} 
                        for office in office_types]
    })


# API Views (when DRF is available)
@api_view(['GET'])
def api_elections(request):
    """API endpoint to list all elections."""
    elections = Election.objects.all()
    data = [{
        'id': e.id,
        'name': e.name,
        'election_type': e.election_type,
        'year': e.year,
        'date': e.date.isoformat() if e.date else None
    } for e in elections]
    
    if DRF_AVAILABLE:
        return Response(data)
    else:
        return JsonResponse({'elections': data})


@api_view(['GET'])
def api_voter_election_data(request, voter_id):
    """API endpoint to get voter's election history."""
    voter = get_object_or_404(VoterRecord, id=voter_id)
    election_data = voter.election_data.all().select_related('election')
    
    data = [{
        'election': {
            'id': ed.election.id,
            'name': ed.election.name,
            'year': ed.election.year,
            'date': ed.election.date.isoformat() if ed.election.date else None
        },
        'data_type': ed.data_type,
        'value': ed.value
    } for ed in election_data]
    
    if DRF_AVAILABLE:
        return Response(data)
    else:
        return JsonResponse({'election_data': data})


@api_view(['GET'])
def api_voter_address(request, voter_id):
    """API endpoint to get voter's address information."""
    voter = get_object_or_404(VoterRecord, id=voter_id)
    verification_status = get_address_verification_status(voter)
    
    if DRF_AVAILABLE:
        return Response(verification_status)
    else:
        return JsonResponse(verification_status)


@api_view(['POST'])
def api_verify_address(request, voter_id):
    """API endpoint to trigger address verification."""
    voter = get_object_or_404(VoterRecord, id=voter_id)
    
    # Start verification task
    verify_address.delay(voter.id)
    
    if DRF_AVAILABLE:
        return Response({'message': 'Address verification started'}, status=status.HTTP_202_ACCEPTED)
    else:
        return JsonResponse({'message': 'Address verification started'}, status=202)


@api_view(['GET'])
def api_voter_district(request, voter_id):
    """API endpoint to get voter's district information."""
    voter = get_object_or_404(VoterRecord, id=voter_id)
    
    data = {
        'district_level': voter.district_level,
        'office_type': voter.office_type,
        'county': voter.county,
        'precinct': voter.pct
    }
    
    if DRF_AVAILABLE:
        return Response(data)
    else:
        return JsonResponse(data)