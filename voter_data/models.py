from django.db import models


class VoterRecord(models.Model):
    """
    Model representing a voter record with personal information, address components,
    and district information.
    """
    # Core voter identification
    vuid = models.CharField(max_length=50, unique=True, help_text="Voter Unique ID")
    
    # Personal information
    lname = models.CharField(max_length=50, verbose_name="Last Name")
    fname = models.CharField(max_length=50, verbose_name="First Name")
    mname = models.CharField(max_length=50, blank=True, verbose_name="Middle Name")
    formername = models.CharField(max_length=50, blank=True, verbose_name="Former Name")
    sfx = models.CharField(max_length=10, blank=True, verbose_name="Suffix")
    sex = models.CharField(max_length=1, blank=True, choices=[('M', 'Male'), ('F', 'Female')])
    dob = models.DateField(null=True, blank=True, verbose_name="Date of Birth")
    edr = models.DateField(null=True, blank=True, verbose_name="Effective Date of Registration")
    status = models.CharField(max_length=20, blank=True, verbose_name="Voter Status")
    
    # Geographic/Political divisions
    county = models.CharField(max_length=50, blank=True)
    pct = models.CharField(max_length=10, blank=True, verbose_name="Precinct")
    
    # District information
    DISTRICT_LEVEL_CHOICES = [
        ('federal', 'Federal'),
        ('state', 'State'),
        ('judicial', 'Judicial'),
        ('county', 'County'),
        ('city', 'City'),
    ]
    district_level = models.CharField(
        max_length=20, 
        choices=DISTRICT_LEVEL_CHOICES, 
        blank=True,
        verbose_name="District Level"
    )
    office_type = models.CharField(max_length=50, blank=True, verbose_name="Office Type")
    
    # Residential address components
    rhnum = models.CharField(max_length=10, blank=True, verbose_name="Residential House Number")
    rdesig = models.CharField(max_length=5, blank=True, verbose_name="Residential Designation")
    rstname = models.CharField(max_length=50, blank=True, verbose_name="Residential Street Name")
    rsttype = models.CharField(max_length=10, blank=True, verbose_name="Residential Street Type")
    rstsf = models.CharField(max_length=20, blank=True, verbose_name="Residential Street Suffix")
    runum = models.CharField(max_length=10, blank=True, verbose_name="Residential Unit Number")
    rutype = models.CharField(max_length=10, blank=True, verbose_name="Residential Unit Type")
    rcity = models.CharField(max_length=50, blank=True, verbose_name="Residential City")
    rzip = models.CharField(max_length=10, blank=True, verbose_name="Residential ZIP")
    
    # Mailing address components
    madr1 = models.CharField(max_length=100, blank=True, verbose_name="Mailing Address Line 1")
    madr2 = models.CharField(max_length=100, blank=True, verbose_name="Mailing Address Line 2")
    mcity = models.CharField(max_length=50, blank=True, verbose_name="Mailing City")
    mst = models.CharField(max_length=2, blank=True, verbose_name="Mailing State")
    mzip = models.CharField(max_length=10, blank=True, verbose_name="Mailing ZIP")
    
    # Constructed addresses
    residential_address = models.TextField(blank=True, verbose_name="Constructed Residential Address")
    mailing_address = models.TextField(blank=True, verbose_name="Constructed Mailing Address")
    
    # Verification and geocoding
    is_verified = models.BooleanField(default=False, verbose_name="Address Verified")
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # File tracking
    file_origin = models.CharField(max_length=50, blank=True, verbose_name="Source File")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Voter Record"
        verbose_name_plural = "Voter Records"
        ordering = ['lname', 'fname']

    def __str__(self):
        return f"{self.fname} {self.lname} ({self.vuid})"

    def get_full_name(self):
        """Return the full name of the voter."""
        parts = [self.fname, self.mname, self.lname, self.sfx]
        return ' '.join(filter(None, parts))


class Election(models.Model):
    """
    Model representing an election with metadata.
    """
    ELECTION_TYPE_CHOICES = [
        ('general', 'General'),
        ('primary', 'Primary'),
        ('special', 'Special'),
        ('referendum', 'Referendum'),
    ]
    
    name = models.CharField(max_length=100, verbose_name="Election Name")
    election_type = models.CharField(
        max_length=50, 
        choices=ELECTION_TYPE_CHOICES, 
        blank=True,
        verbose_name="Election Type"
    )
    year = models.PositiveIntegerField(blank=True, null=True, verbose_name="Election Year")
    date = models.DateField(blank=True, null=True, verbose_name="Election Date")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Election"
        verbose_name_plural = "Elections"
        ordering = ['-date', '-year', 'name']

    def __str__(self):
        if self.year:
            return f"{self.name} ({self.year})"
        return self.name


class ElectionData(models.Model):
    """
    Model linking voters to election-specific data.
    """
    DATA_TYPE_CHOICES = [
        ('location', 'Voting Location'),
        ('ballot_type', 'Ballot Type'),
        ('ballot_choice', 'Ballot Choice'),
        ('voted', 'Voted (Boolean)'),
    ]
    
    voter = models.ForeignKey(
        VoterRecord, 
        on_delete=models.CASCADE, 
        related_name='election_data',
        verbose_name="Voter"
    )
    election = models.ForeignKey(
        Election, 
        on_delete=models.CASCADE, 
        related_name='voter_data',
        verbose_name="Election"
    )
    data_type = models.CharField(
        max_length=50, 
        choices=DATA_TYPE_CHOICES,
        verbose_name="Data Type"
    )
    value = models.TextField(verbose_name="Data Value")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Election Data"
        verbose_name_plural = "Election Data"
        unique_together = ['voter', 'election', 'data_type']
        ordering = ['-election__date', 'voter__lname', 'voter__fname']

    def __str__(self):
        return f"{self.voter} - {self.election} ({self.data_type})"