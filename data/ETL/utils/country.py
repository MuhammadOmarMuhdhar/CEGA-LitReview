import re
import pycountry
from typing import List, Dict, Set
import spacy
from collections import Counter

class Country:
    """
    Fast country extraction using multiple methods for research abstracts.
    """
    
    def __init__(self):
        """Initialize the country extractor with various lookup methods."""
        self.country_patterns = self._build_country_patterns()
        self.city_to_country = self._build_city_mappings()
        
        # Try to load spaCy model (optional, fallback to regex if not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            print("Warning: spaCy model not found. Using regex-only approach.")
            self.nlp = None
            self.use_spacy = False
    
    def _build_country_patterns(self) -> Dict[str, str]:
        """Build comprehensive country name patterns including variations."""
        patterns = {}
        
        for country in pycountry.countries:
            # Official name from pycountry
            patterns[country.name.lower()] = country.name
            
            # Add official_name if available
            if hasattr(country, 'official_name') and country.official_name:
                patterns[country.official_name.lower()] = country.name
            
            # Add alpha codes as text variations  
            patterns[country.alpha_2.lower()] = country.name
            patterns[country.alpha_3.lower()] = country.name
        
        # Manual variations that pycountry doesn't provide
        # These are common informal names found in research papers
        manual_variations = {
            'united states': ['usa', 'us', 'america', 'united states of america'],
            'united kingdom': ['uk', 'britain', 'great britain', 'england', 'scotland', 'wales', 'northern ireland'],
            'south korea': ['korea', 'republic of korea', 'south korea'],
            'north korea': ['dprk', 'democratic people\'s republic of korea'],
            'russia': ['russian federation', 'ussr', 'soviet union'],
            'iran': ['islamic republic of iran', 'persia'],
            'vietnam': ['viet nam'],
            'czech republic': ['czechia', 'czechoslovakia'],
            'democratic republic of the congo': ['drc', 'congo', 'zaire'],
            'republic of the congo': ['congo-brazzaville'],
            'ivory coast': ['côte d\'ivoire'],
            'bosnia and herzegovina': ['bosnia'],
            'trinidad and tobago': ['trinidad'],
            'antigua and barbuda': ['antigua'],
            'saint vincent and the grenadines': ['saint vincent'],
            'sao tome and principe': ['sao tome'],
            'myanmar': ['burma'],
            'netherlands': ['holland'],
            'switzerland': ['swiss confederation'],
            'vatican city': ['holy see'],
        }
        
        # Add manual variations to patterns
        for standard_name, variations in manual_variations.items():
            for variant in variations:
                patterns[variant.lower()] = self._get_standard_country_name(standard_name)
        
        return patterns
    
    def _get_standard_country_name(self, country_name_lower: str) -> str:
        """Get the standard pycountry name for a country."""
        # Try to find the country in pycountry
        for country in pycountry.countries:
            if country.name.lower() == country_name_lower:
                return country.name
        
        # Fallback to title case if not found
        return country_name_lower.title()
    
    def _build_city_mappings(self) -> Dict[str, str]:
        """Build mappings from major cities to countries."""
        # This is a simplified version - in practice, you'd want a comprehensive database
        city_mappings = {
            # Major cities that frequently appear in research
            'new york': 'United States',
            'los angeles': 'United States',
            'chicago': 'United States',
            'boston': 'United States',
            'washington': 'United States',
            'london': 'United Kingdom',
            'manchester': 'United Kingdom',
            'birmingham': 'United Kingdom',
            'paris': 'France',
            'lyon': 'France',
            'marseille': 'France',
            'berlin': 'Germany',
            'munich': 'Germany',
            'hamburg': 'Germany',
            'tokyo': 'Japan',
            'osaka': 'Japan',
            'kyoto': 'Japan',
            'beijing': 'China',
            'shanghai': 'China',
            'guangzhou': 'China',
            'delhi': 'India',
            'mumbai': 'India',
            'bangalore': 'India',
            'kolkata': 'India',
            'toronto': 'Canada',
            'vancouver': 'Canada',
            'montreal': 'Canada',
            'sydney': 'Australia',
            'melbourne': 'Australia',
            'brisbane': 'Australia',
            'seoul': 'South Korea',
            'moscow': 'Russia',
            'st petersburg': 'Russia',
            'cairo': 'Egypt',
            'nairobi': 'Kenya',
            'lagos': 'Nigeria',
            'cape town': 'South Africa',
            'johannesburg': 'South Africa',
            'sao paulo': 'Brazil',
            'rio de janeiro': 'Brazil',
            'mexico city': 'Mexico',
            'buenos aires': 'Argentina',
        }
        return city_mappings
    
    def extract_countries_regex(self, text: str) -> List[str]:
        """Extract countries using regex pattern matching."""
        text_lower = text.lower()
        found_countries = set()
        
        # Look for country patterns
        for pattern, country_name in self.country_patterns.items():
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                found_countries.add(country_name)
        
        # Look for city patterns
        for city, country in self.city_to_country.items():
            if re.search(r'\b' + re.escape(city) + r'\b', text_lower):
                found_countries.add(country)
        
        return list(found_countries)
    
    def extract_countries_spacy(self, text: str) -> List[str]:
        """Extract countries using spaCy NER."""
        if not self.use_spacy:
            return []
        
        doc = self.nlp(text)
        found_countries = set()
        
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entity or location
                entity_text = ent.text.lower()
                if entity_text in self.country_patterns:
                    found_countries.add(self.country_patterns[entity_text])
                elif entity_text in self.city_to_country:
                    found_countries.add(self.city_to_country[entity_text])
        
        return list(found_countries)
    
    def extract_countries_combined(self, text: str) -> List[str]:
        """Combine regex and spaCy approaches for best results."""
        countries_regex = self.extract_countries_regex(text)
        countries_spacy = self.extract_countries_spacy(text) if self.use_spacy else []
        
        # Combine and deduplicate
        all_countries = list(set(countries_regex + countries_spacy))
        
        return all_countries
    
    def extract(self, abstract: str, method: str = 'combined') -> str:
        """
        Extract countries from research abstract and return as comma-separated string.
        
        Args:
            abstract: The research abstract text
            method: 'regex', 'spacy', or 'combined'
        
        Returns:
            Comma-separated country names or "Insufficient info"
        """
        if method == 'regex':
            countries = self.extract_countries_regex(abstract)
        elif method == 'spacy':
            countries = self.extract_countries_spacy(abstract)
        else:
            countries = self.extract_countries_combined(abstract)
        
        if countries:
            return ', '.join(sorted(countries))
        else:
            return "Insufficient info"
