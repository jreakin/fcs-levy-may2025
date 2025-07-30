from django.contrib import admin
from .models import VoterRecord, Election, ElectionData


@admin.register(VoterRecord)
class VoterRecordAdmin(admin.ModelAdmin):
    list_display = ['vuid', 'fname', 'lname', 'county', 'pct', 'district_level', 'office_type', 'is_verified']
    list_filter = ['county', 'district_level', 'office_type', 'is_verified', 'status']
    search_fields = ['vuid', 'fname', 'lname', 'county']
    readonly_fields = ['created_at', 'updated_at', 'residential_address', 'mailing_address']
    
    fieldsets = (
        ('Personal Information', {
            'fields': ('vuid', 'fname', 'mname', 'lname', 'formername', 'sfx', 'sex', 'dob', 'edr', 'status')
        }),
        ('District Information', {
            'fields': ('county', 'pct', 'district_level', 'office_type')
        }),
        ('Residential Address Components', {
            'fields': ('rhnum', 'rdesig', 'rstname', 'rsttype', 'rstsf', 'runum', 'rutype', 'rcity', 'rzip')
        }),
        ('Mailing Address Components', {
            'fields': ('madr1', 'madr2', 'mcity', 'mst', 'mzip')
        }),
        ('Constructed Addresses', {
            'fields': ('residential_address', 'mailing_address'),
            'classes': ('collapse',)
        }),
        ('Verification', {
            'fields': ('is_verified', 'latitude', 'longitude')
        }),
        ('File Information', {
            'fields': ('file_origin', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Election)
class ElectionAdmin(admin.ModelAdmin):
    list_display = ['name', 'election_type', 'year', 'date']
    list_filter = ['election_type', 'year']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(ElectionData)
class ElectionDataAdmin(admin.ModelAdmin):
    list_display = ['voter', 'election', 'data_type', 'value']
    list_filter = ['election', 'data_type']
    search_fields = ['voter__fname', 'voter__lname', 'voter__vuid', 'election__name']
    readonly_fields = ['created_at', 'updated_at']