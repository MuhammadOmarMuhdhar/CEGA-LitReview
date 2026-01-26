"""
Statistics Cache Generator

Generates pre-computed statistics and filter combinations to eliminate
BigQuery queries for geography filters and statistics display.

This reduces page load times from 5-7 seconds to <100ms.
"""

import json
import os
import ast
import logging
from datetime import datetime
from collections import Counter
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticsCacheGenerator:
    """Generates pre-computed statistics cache from BigQuery data"""

    def __init__(self, bigquery_client, project_id='literature-452020'):
        """
        Initialize cache generator

        Args:
            bigquery_client: Instance of data.bigQuery.Client
            project_id: BigQuery project ID
        """
        self.client = bigquery_client
        self.project_id = project_id
        self.cache_path = "data/cache/statistics_cache.json"
        self.dataset_id = "psychology_of_poverty_literature"
        self.table_id = "papers"

    def extract_filters(self):
        """
        Query BigQuery for all unique countries and institutions

        Returns:
            tuple: (countries_list, institutions_list, institutions_by_country_dict)
        """
        logger.info("Extracting filters from BigQuery...")

        query = f"""
            SELECT DISTINCT
                country_of_study,
                institution
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE country_of_study IS NOT NULL
              AND institution IS NOT NULL
              AND country_of_study != ''
              AND institution != ''
        """

        result = self.client.execute_query(query)

        countries = set()
        institutions = set()
        institutions_by_country = {}

        for _, row in result.iterrows():
            # Parse countries (comma-separated)
            country_list = [c.strip() for c in str(row['country_of_study']).split(',')
                           if c.strip() and c.strip().lower() != 'nan']
            countries.update(country_list)

            # Parse institutions (list format or comma-separated)
            try:
                inst_list = ast.literal_eval(str(row['institution']))
                if isinstance(inst_list, list):
                    inst_list = [str(i) for i in inst_list if i]
                else:
                    inst_list = []
            except:
                # Fallback for non-list format
                inst_list = [i.strip() for i in str(row['institution']).split(',')
                            if i.strip()]

            institutions.update(inst_list)

            # Build institutions_by_country mapping
            for country in country_list:
                if country not in institutions_by_country:
                    institutions_by_country[country] = set()
                institutions_by_country[country].update(inst_list)

        # Convert sets to sorted lists
        countries_list = sorted(list(countries))
        institutions_list = sorted(list(institutions))
        institutions_by_country_dict = {
            country: sorted(list(insts))
            for country, insts in institutions_by_country.items()
        }

        logger.info(f"Extracted {len(countries_list)} countries, {len(institutions_list)} institutions")

        return countries_list, institutions_list, institutions_by_country_dict

    def compute_statistics(self, country='All', institution='All'):
        """
        Compute statistics for one filter combination

        Args:
            country: Country filter ('All' or specific country)
            institution: Institution filter ('All' or specific institution)

        Returns:
            dict: Statistics for this filter combination
        """
        logger.info(f"Computing statistics for {country} / {institution}")

        # Build WHERE clause
        where_conditions = []
        if country != 'All':
            safe_country = country.replace("'", "\\'")
            where_conditions.append(f"REGEXP_CONTAINS(country_of_study, r'\\b{safe_country}\\b')")

        if institution != 'All':
            safe_institution = institution.replace("'", "\\'")
            where_conditions.append(f"REGEXP_CONTAINS(institution, r'\\b{safe_institution}\\b')")

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Query data
        query = f"""
            SELECT doi, date, institution, country_of_study
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE {where_clause}
        """

        df = self.client.execute_query(query)

        if df.empty:
            return None

        # Compute statistics
        total_papers = len(df)

        # Date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = f"{min_date} – {max_date}"

        # Count unique countries
        all_countries = set()
        for _, row in df.iterrows():
            country_list = [c.strip() for c in str(row['country_of_study']).split(',')
                           if c.strip() and c.strip().lower() != 'nan']
            all_countries.update(country_list)
        countries_count = len(all_countries)

        # Count unique institutions and build top 10
        all_institutions = []
        for _, row in df.iterrows():
            try:
                inst_list = ast.literal_eval(str(row['institution']))
                if isinstance(inst_list, list):
                    all_institutions.extend([str(i) for i in inst_list if i])
            except:
                inst_list = [i.strip() for i in str(row['institution']).split(',')]
                all_institutions.extend([i for i in inst_list if i])

        institutions_count = len(set(all_institutions))

        # Top 10 institutions for bar chart
        institution_counts = Counter(all_institutions)
        top_institutions = [
            {'institution': inst, 'count': count}
            for inst, count in institution_counts.most_common(10)
        ]

        return {
            'total_papers': total_papers,
            'date_range': date_range,
            'min_date': str(min_date),
            'max_date': str(max_date),
            'countries_count': countries_count,
            'institutions_count': institutions_count,
            'top_institutions': top_institutions
        }

    def generate_cache(self):
        """
        Generate complete cache structure

        Returns:
            dict: Complete cache data structure
        """
        start_time = datetime.now()
        logger.info("Starting cache generation...")

        # Phase 1: Extract all filters
        countries, institutions, institutions_by_country = self.extract_filters()

        # Phase 2: Compute statistics for all combinations
        statistics = {}
        total_combinations = 1 + len(countries) + sum(len(insts) for insts in institutions_by_country.values())
        current = 0

        # Global stats (All/All)
        logger.info(f"[{current+1}/{total_combinations}] Computing global statistics...")
        stats = self.compute_statistics("All", "All")
        if stats:
            statistics["All__All"] = stats
        current += 1

        # Per-country stats
        for country in countries:
            current += 1
            logger.info(f"[{current}/{total_combinations}] Computing stats for {country}...")
            stats = self.compute_statistics(country, "All")
            if stats and stats['total_papers'] > 0:
                statistics[f"{country}__All"] = stats

        # Per-country-institution stats
        for country, insts in institutions_by_country.items():
            for institution in insts:
                current += 1
                logger.info(f"[{current}/{total_combinations}] Computing stats for {country} / {institution}...")
                stats = self.compute_statistics(country, institution)
                if stats and stats['total_papers'] > 0:
                    statistics[f"{country}__{institution}"] = stats

        # Calculate generation duration
        duration = (datetime.now() - start_time).total_seconds()

        # Build final structure
        cache_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat() + 'Z',
                'schema_version': '1.0',
                'total_papers': statistics["All__All"]['total_papers'],
                'generation_duration_seconds': round(duration, 2)
            },
            'filters': {
                'countries': countries,
                'institutions': institutions,
                'institutions_by_country': institutions_by_country
            },
            'statistics': statistics
        }

        logger.info(f"Cache generation completed in {duration:.2f} seconds")
        logger.info(f"Total combinations: {len(statistics)}")

        return cache_data

    def run(self):
        """
        Main entry point - generates and saves cache to file

        Returns:
            dict: Generated cache data
        """
        logger.info("=" * 60)
        logger.info("Statistics Cache Generator")
        logger.info("=" * 60)

        # Generate cache
        cache_data = self.generate_cache()

        # Write to file
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Cache saved to: {self.cache_path}")

        # Calculate file size
        file_size = os.path.getsize(self.cache_path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Cache file size: {file_size_mb:.2f} MB")

        logger.info("=" * 60)

        return cache_data
