# Django Political Campaign Management System

This Django application extends the existing voter analysis system with comprehensive campaign management features including election data handling, address verification, and district management.

## Features

### 1. Election Data Handling
- **Generic Election Fields**: Specify election metadata for any CSV column during upload
- **Flexible Data Types**: Support for voting locations, ballot types, ballot choices, and voted boolean
- **Election Management**: Store and manage election metadata with type, year, and date information

### 2. Address Parsing and Verification
- **Address Construction**: Build complete addresses from fragmented components (RHNUM, RDESIG, RSTNAME, etc.)
- **Geocoding**: Verify addresses using Nominatim geocoding service
- **Validation**: Cross-check addresses with county and precinct data for consistency
- **Status Tracking**: Mark addresses as verified/unverified with latitude/longitude coordinates

### 3. District Data Management
- **Hierarchical Districts**: Federal, State, Judicial, County, and City levels
- **Dynamic Office Types**: Office options update based on selected district level
- **Validation**: Ensure office types are appropriate for district levels

## Installation

### Prerequisites
- Python 3.12+
- Django 4.2+
- Redis (for Celery tasks)
- PostgreSQL (for production, SQLite for development)

### Setup Steps

1. **Install Dependencies**:
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # OR using UV (if available)
   uv sync
   ```

2. **Configure Database**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

3. **Create Superuser**:
   ```bash
   python manage.py createsuperuser
   ```

4. **Run Development Server**:
   ```bash
   python manage.py runserver
   ```

5. **Start Celery Worker** (for address verification):
   ```bash
   celery -A campaign_management worker --loglevel=info
   ```

## Usage

### Web Interface

1. **Home Page**: Navigate to `http://localhost:8000/` for the main dashboard
2. **Upload Data**: Use the upload form to process CSV files with voter and election data
3. **Browse Voters**: View and search voter records with address verification status
4. **Manage Elections**: View election metadata and associated voter data

### CSV Upload Process

1. **Upload File**: Select a CSV file with voter data
2. **Specify Elections**: Define election metadata for any columns containing election data
3. **Set Districts**: Choose district level and office type for the upload batch
4. **Process Data**: System will parse addresses and start verification tasks

### Expected CSV Format

Your CSV should include:

**Required Fields**:
- `VUID` - Voter Unique ID
- `FNAME`, `LNAME` - Voter names
- `COUNTY`, `PCT` - Geographic identifiers

**Address Fields**:
- `RHNUM`, `RDESIG`, `RSTNAME`, `RSTTYPE`, `RSTSF` - Residential address components
- `RUNUM`, `RUTYPE` - Residential unit information
- `RCITY`, `RZIP` - Residential city and ZIP
- `MADR1`, `MADR2`, `MCITY`, `MST`, `MZIP` - Mailing address

**Election Fields**: Any additional columns can be specified as election data

## API Endpoints

### Election Data
- `GET /api/elections/` - List all elections
- `GET /api/voters/{id}/election-data/` - Get voter's election history

### Address Management
- `GET /api/voters/{id}/address/` - Get voter's address information
- `POST /api/voters/{id}/verify-address/` - Trigger address verification

### District Management
- `GET /api/voters/{id}/district/` - Get voter's district information

## Configuration

### Settings

Key settings in `campaign_management/settings.py`:

```python
# Celery Configuration
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'django-db'

# Geocoding settings
GEOCODING_USER_AGENT = 'campaign_management'

# Database (for production)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'campaign_db',
        'USER': 'db_user',
        'PASSWORD': 'db_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Environment Variables

For production deployment:

```bash
export DJANGO_SETTINGS_MODULE=campaign_management.settings
export DATABASE_URL=postgresql://user:pass@localhost/dbname
export CELERY_BROKER_URL=redis://localhost:6379
export SECRET_KEY=your-secret-key
export DEBUG=False
```

## Models

### VoterRecord
Stores comprehensive voter information including personal data, address components, district information, and verification status.

### Election
Manages election metadata including name, type, year, and date.

### ElectionData
Links voters to specific election data with flexible data types.

## Address Verification

The system uses asynchronous tasks to verify addresses:

1. **Address Construction**: Combines fragmented address components into complete addresses
2. **Geocoding**: Uses Nominatim to get latitude/longitude coordinates
3. **Validation**: Cross-checks with existing county/precinct data
4. **Status Updates**: Marks addresses as verified or unverified

## District Management

Dynamic office type selection based on district level:

- **Federal**: President, Senate, House of Representatives
- **State**: Governor, State Senate, State House, Attorney General
- **Judicial**: Supreme Court Justice, District Judge
- **County**: County Commissioner, Sheriff, County Clerk
- **City**: Mayor, City Council, City Treasurer

## Testing

Run the test suite:

```bash
python manage.py test voter_data
```

Tests cover:
- Model functionality
- Address construction utilities
- District validation
- View responses
- API endpoints

## Integration with Existing System

This Django application coexists with the existing voter analysis codebase:

- **Data Analysis**: Original ML models and analysis tools remain in `src/fcs_may25/`
- **Web Interface**: New Django app provides web-based data management
- **Shared Data**: Both systems can work with the same voter data
- **API Access**: RESTful endpoints allow integration with external tools

## Security Considerations

- **CSRF Protection**: All forms include CSRF tokens
- **Input Validation**: Address and district data is validated
- **Authentication**: API endpoints can be protected with authentication
- **Data Privacy**: Voter data handling follows privacy best practices

## Deployment

For production deployment:

1. **Use PostgreSQL**: Configure production database
2. **Configure Redis**: Set up Redis for Celery tasks
3. **Set Environment Variables**: Configure production settings
4. **Collect Static Files**: `python manage.py collectstatic`
5. **Use WSGI Server**: Deploy with Gunicorn or similar
6. **Set Up Celery**: Configure Celery workers for background tasks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is part of the FCS Levy Analysis system. Please refer to the main project license.