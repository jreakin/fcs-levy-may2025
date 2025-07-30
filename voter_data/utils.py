from typing import Tuple
from .models import VoterRecord


def construct_address_lines(voter: VoterRecord) -> Tuple[str, str]:
    """
    Construct full address lines from fragmented address components.
    
    Args:
        voter: VoterRecord instance with address components
        
    Returns:
        Tuple of (residential_address, mailing_address)
    """
    # Construct residential address
    res_components = [
        voter.rhnum or "",
        voter.rdesig or "",
        voter.rstname or "",
        voter.rsttype or ""
    ]
    res_address = " ".join(filter(None, res_components)).strip()
    
    # Add unit information if present
    if voter.runum and voter.rutype:
        res_address += f" {voter.rutype} {voter.runum}"
    
    # Add suffix if present
    if voter.rstsf:
        res_address += f", {voter.rstsf}"
    
    # Construct city, state, zip for residential
    res_city_state_zip = f"{voter.rcity or ''}, {voter.rzip or ''}".strip().strip(",")
    
    # Construct mailing address
    mail_address = f"{voter.madr1 or ''} {voter.madr2 or ''}".strip()
    mail_city_state_zip = f"{voter.mcity or ''}, {voter.mst or ''} {voter.mzip or ''}".strip().strip(",")
    
    # Return full addresses
    residential_full = f"{res_address}\n{res_city_state_zip}" if res_city_state_zip else res_address
    mailing_full = f"{mail_address}\n{mail_city_state_zip}" if mail_city_state_zip and mail_address else ""
    
    return residential_full.strip(), mailing_full.strip()


def update_addresses(voter: VoterRecord) -> VoterRecord:
    """
    Update the constructed address fields for a voter record.
    
    Args:
        voter: VoterRecord instance to update
        
    Returns:
        Updated VoterRecord instance
    """
    residential_addr, mailing_addr = construct_address_lines(voter)
    voter.residential_address = residential_addr
    voter.mailing_address = mailing_addr
    voter.save()
    return voter


def get_office_types_for_district_level(district_level: str) -> list:
    """
    Get the available office types for a given district level.
    
    Args:
        district_level: The district level (federal, state, judicial, county, city)
        
    Returns:
        List of office type options
    """
    office_types = {
        'federal': ['President', 'Senate', 'House of Representatives'],
        'state': ['Governor', 'State Senate', 'State House', 'Attorney General'],
        'judicial': ['Supreme Court Justice', 'District Judge'],
        'county': ['County Commissioner', 'Sheriff', 'County Clerk'],
        'city': ['Mayor', 'City Council', 'City Treasurer']
    }
    
    return office_types.get(district_level, [])


def validate_district_and_office(district_level: str, office_type: str) -> bool:
    """
    Validate that an office type is valid for the given district level.
    
    Args:
        district_level: The district level
        office_type: The office type to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_offices = get_office_types_for_district_level(district_level)
    return office_type in valid_offices


def parse_csv_header_for_elections(header: list) -> list:
    """
    Parse CSV header and identify potential election columns.
    
    Args:
        header: List of column names from CSV
        
    Returns:
        List of column names that could contain election data
    """
    # Look for columns that might contain election data
    # This could be any column based on user specification
    potential_election_columns = []
    
    # Common patterns that might indicate election data
    election_patterns = [
        'GEN', 'PRI', 'ELECTION', 'VOTE', 'BALLOT', 'LOCATION',
        'SPECIAL', 'REFERENDUM', 'TURNOUT', 'HISTORY'
    ]
    
    for col in header:
        col_upper = col.upper()
        # Check if column matches common election patterns
        if any(pattern in col_upper for pattern in election_patterns):
            potential_election_columns.append(col)
        # Also include any column that user might want to specify as election data
        elif len(col) > 0:  # Any non-empty column could potentially be election data
            potential_election_columns.append(col)
    
    return potential_election_columns


def clean_and_validate_voter_data(data_dict: dict) -> dict:
    """
    Clean and validate voter data from CSV input.
    
    Args:
        data_dict: Dictionary of voter data from CSV row
        
    Returns:
        Cleaned and validated data dictionary
    """
    # Map common CSV column names to model field names
    field_mapping = {
        'VUID': 'vuid',
        'LNAME': 'lname',
        'FNAME': 'fname',
        'MNAME': 'mname',
        'FORMERNAME': 'formername',
        'SFX': 'sfx',
        'SEX': 'sex',
        'DOB': 'dob',
        'EDR': 'edr',
        'STATUS': 'status',
        'COUNTY': 'county',
        'PCT': 'pct',
        'RHNUM': 'rhnum',
        'RDESIG': 'rdesig',
        'RSTNAME': 'rstname',
        'RSTTYPE': 'rsttype',
        'RSTSFX': 'rstsf',
        'RUNUM': 'runum',
        'RUTYPE': 'rutype',
        'RCITY': 'rcity',
        'RZIP': 'rzip',
        'MADR1': 'madr1',
        'MADR2': 'madr2',
        'MCITY': 'mcity',
        'MST': 'mst',
        'MZIP': 'mzip',
    }
    
    cleaned_data = {}
    
    for csv_field, model_field in field_mapping.items():
        if csv_field in data_dict:
            value = data_dict[csv_field]
            # Clean the value
            if isinstance(value, str):
                value = value.strip()
                if value == '':
                    value = None
            cleaned_data[model_field] = value
    
    return cleaned_data