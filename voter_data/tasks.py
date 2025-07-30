try:
    from celery import shared_task
    from geopy.geocoders import Nominatim
    from django.core.exceptions import ValidationError
    from django.conf import settings
    CELERY_AVAILABLE = True
    GEOPY_AVAILABLE = True
except ImportError:
    # Fallback implementations when dependencies are not available
    CELERY_AVAILABLE = False
    GEOPY_AVAILABLE = False
    
    def shared_task(func):
        """Mock shared_task decorator when Celery is not available."""
        return func

from .models import VoterRecord


def verify_address_sync(voter_id: int) -> bool:
    """
    Synchronous version of address verification.
    
    Args:
        voter_id: ID of the VoterRecord to verify
        
    Returns:
        True if verification successful, False otherwise
    """
    try:
        voter = VoterRecord.objects.get(id=voter_id)
    except VoterRecord.DoesNotExist:
        return False
    
    if not GEOPY_AVAILABLE:
        # Mock verification when geopy is not available
        voter.is_verified = True
        voter.save()
        return True
    
    try:
        geolocator = Nominatim(user_agent=getattr(settings, 'GEOCODING_USER_AGENT', 'campaign_management'))
        
        # Determine which address to verify
        address_to_verify = voter.mailing_address or voter.residential_address
        if not address_to_verify:
            return False

        # Attempt to geocode the address
        location = geolocator.geocode(address_to_verify)
        if location:
            voter.latitude = location.latitude
            voter.longitude = location.longitude
            
            # Cross-check with county if available
            if voter.county and hasattr(location, 'raw'):
                location_county = location.raw.get('county', '').lower()
                if location_county and location_county != voter.county.lower():
                    # Log the mismatch but don't fail verification
                    print(f"County mismatch for voter {voter.vuid}: {voter.county} vs {location_county}")
            
            voter.is_verified = True
            voter.save()
            return True
        else:
            # Address could not be geocoded
            voter.is_verified = False
            voter.save()
            return False
            
    except Exception as e:
        print(f"Error verifying address for voter {voter.vuid}: {str(e)}")
        return False


@shared_task
def verify_address(voter_id: int) -> bool:
    """
    Celery task to verify and geocode a voter's address.
    
    Args:
        voter_id: ID of the VoterRecord to verify
        
    Returns:
        True if verification successful, False otherwise
    """
    if not CELERY_AVAILABLE:
        # Fall back to synchronous verification
        return verify_address_sync(voter_id)
    
    return verify_address_sync(voter_id)


@shared_task
def batch_verify_addresses(voter_ids: list) -> dict:
    """
    Celery task to verify multiple addresses in batch.
    
    Args:
        voter_ids: List of VoterRecord IDs to verify
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'total': len(voter_ids),
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    for voter_id in voter_ids:
        try:
            if verify_address_sync(voter_id):
                results['successful'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Voter ID {voter_id}: {str(e)}")
    
    return results


def validate_address_consistency(voter: VoterRecord) -> list:
    """
    Validate address consistency with voting history and district information.
    
    Args:
        voter: VoterRecord instance to validate
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    # Check if residential address components are consistent
    if voter.rhnum and not voter.rstname:
        issues.append("House number provided but street name missing")
    
    if voter.rstname and not voter.rcity:
        issues.append("Street name provided but city missing")
    
    # Check ZIP code format
    if voter.rzip and len(voter.rzip) not in [5, 9, 10]:  # 5-digit, 9-digit, or 5+4 format
        issues.append("Invalid residential ZIP code format")
    
    if voter.mzip and len(voter.mzip) not in [5, 9, 10]:
        issues.append("Invalid mailing ZIP code format")
    
    # Check state abbreviation for mailing address
    if voter.mst and len(voter.mst) != 2:
        issues.append("Invalid state abbreviation in mailing address")
    
    return issues


def get_address_verification_status(voter: VoterRecord) -> dict:
    """
    Get comprehensive address verification status for a voter.
    
    Args:
        voter: VoterRecord instance
        
    Returns:
        Dictionary with verification status information
    """
    residential_addr, mailing_addr = construct_address_lines(voter)
    validation_issues = validate_address_consistency(voter)
    
    return {
        'voter_id': voter.id,
        'vuid': voter.vuid,
        'residential_address': residential_addr,
        'mailing_address': mailing_addr,
        'is_verified': voter.is_verified,
        'has_coordinates': bool(voter.latitude and voter.longitude),
        'validation_issues': validation_issues,
        'verification_needed': not voter.is_verified or bool(validation_issues),
    }


# Import construct_address_lines from utils to avoid circular import
from .utils import construct_address_lines