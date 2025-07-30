from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from .models import VoterRecord, Election, ElectionData
from .utils import construct_address_lines, get_office_types_for_district_level, validate_district_and_office


class VoterRecordModelTest(TestCase):
    """Test cases for VoterRecord model."""
    
    def setUp(self):
        self.voter = VoterRecord.objects.create(
            vuid='TEST001',
            fname='John',
            lname='Doe',
            county='Test County',
            pct='001',
            rhnum='123',
            rstname='Main',
            rsttype='St',
            rcity='Testville',
            rzip='12345'
        )
    
    def test_voter_creation(self):
        """Test voter record creation."""
        self.assertEqual(self.voter.vuid, 'TEST001')
        self.assertEqual(self.voter.fname, 'John')
        self.assertEqual(self.voter.lname, 'Doe')
    
    def test_get_full_name(self):
        """Test full name generation."""
        self.assertEqual(self.voter.get_full_name(), 'John Doe')
        
        # Test with middle name and suffix
        self.voter.mname = 'William'
        self.voter.sfx = 'Jr'
        self.assertEqual(self.voter.get_full_name(), 'John William Doe Jr')
    
    def test_string_representation(self):
        """Test string representation."""
        expected = "John Doe (TEST001)"
        self.assertEqual(str(self.voter), expected)


class ElectionModelTest(TestCase):
    """Test cases for Election model."""
    
    def setUp(self):
        self.election = Election.objects.create(
            name='2024 General Election',
            election_type='general',
            year=2024
        )
    
    def test_election_creation(self):
        """Test election creation."""
        self.assertEqual(self.election.name, '2024 General Election')
        self.assertEqual(self.election.election_type, 'general')
        self.assertEqual(self.election.year, 2024)
    
    def test_string_representation(self):
        """Test string representation."""
        expected = "2024 General Election (2024)"
        self.assertEqual(str(self.election), expected)


class UtilsTest(TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        self.voter = VoterRecord.objects.create(
            vuid='TEST002',
            fname='Jane',
            lname='Smith',
            rhnum='456',
            rdesig='N',
            rstname='Oak',
            rsttype='Ave',
            runum='2B',
            rutype='Apt',
            rcity='Testtown',
            rzip='54321',
            madr1='PO Box 123',
            mcity='Mailtown',
            mst='ST',
            mzip='67890'
        )
    
    def test_construct_address_lines(self):
        """Test address construction from components."""
        residential, mailing = construct_address_lines(self.voter)
        
        # Check residential address
        self.assertIn('456 N Oak Ave Apt 2B', residential)
        self.assertIn('Testtown, 54321', residential)
        
        # Check mailing address
        self.assertIn('PO Box 123', mailing)
        self.assertIn('Mailtown, ST 67890', mailing)
    
    def test_get_office_types_for_district_level(self):
        """Test office type retrieval for district levels."""
        federal_offices = get_office_types_for_district_level('federal')
        self.assertIn('President', federal_offices)
        self.assertIn('Senate', federal_offices)
        
        state_offices = get_office_types_for_district_level('state')
        self.assertIn('Governor', state_offices)
        
        # Test invalid district level
        invalid_offices = get_office_types_for_district_level('invalid')
        self.assertEqual(invalid_offices, [])
    
    def test_validate_district_and_office(self):
        """Test district and office type validation."""
        # Valid combinations
        self.assertTrue(validate_district_and_office('federal', 'President'))
        self.assertTrue(validate_district_and_office('state', 'Governor'))
        
        # Invalid combinations
        self.assertFalse(validate_district_and_office('federal', 'Governor'))
        self.assertFalse(validate_district_and_office('state', 'President'))


class ViewsTest(TestCase):
    """Test cases for views."""
    
    def setUp(self):
        self.client = Client()
        self.voter = VoterRecord.objects.create(
            vuid='TEST003',
            fname='Bob',
            lname='Johnson',
            county='Test County'
        )
    
    def test_home_view(self):
        """Test home page view."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Political Campaign Management System')
    
    def test_voter_list_view(self):
        """Test voter list view."""
        response = self.client.get(reverse('voter_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Bob Johnson')
    
    def test_voter_detail_view(self):
        """Test voter detail view."""
        response = self.client.get(reverse('voter_detail', args=[self.voter.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Bob Johnson')
        self.assertContains(response, 'TEST003')
    
    def test_upload_voter_file_get(self):
        """Test voter file upload form display."""
        response = self.client.get(reverse('upload_voter_file'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Upload Voter File')
    
    def test_election_list_view(self):
        """Test election list view."""
        election = Election.objects.create(
            name='Test Election',
            election_type='general',
            year=2024
        )
        
        response = self.client.get(reverse('election_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Election')


class APITest(TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        self.client = Client()
        self.election = Election.objects.create(
            name='API Test Election',
            election_type='primary',
            year=2024
        )
        self.voter = VoterRecord.objects.create(
            vuid='API001',
            fname='API',
            lname='User',
            county='API County'
        )
    
    def test_api_elections(self):
        """Test elections API endpoint."""
        response = self.client.get(reverse('api_elections'))
        self.assertEqual(response.status_code, 200)
        
        # Check if it's JSON response
        if response.get('Content-Type') == 'application/json':
            data = response.json()
            self.assertIn('elections', data)
    
    def test_api_voter_address(self):
        """Test voter address API endpoint."""
        response = self.client.get(reverse('api_voter_address', args=[self.voter.id]))
        self.assertEqual(response.status_code, 200)
    
    def test_api_voter_district(self):
        """Test voter district API endpoint."""
        response = self.client.get(reverse('api_voter_district', args=[self.voter.id]))
        self.assertEqual(response.status_code, 200)