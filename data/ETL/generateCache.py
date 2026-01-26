import sys
import os
import json
import ast
import logging
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.bigQuery import Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Statistics Cache Generator")
    logger.info("=" * 60)

    # Load credentials
    load_dotenv()
    credentials = {
        "type": os.getenv("type"),
        "project_id": os.getenv("project_id"),
        "private_key_id": os.getenv("private_key_id"),
        "private_key": os.getenv("private_key").replace("\\n", "\n") if os.getenv("private_key") else None,
        "client_email": os.getenv("client_email"),
        "client_id": os.getenv("client_id"),
        "auth_uri": os.getenv("auth_uri"),
        "token_uri": os.getenv("token_uri"),
        "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
        "client_x509_cert_url": os.getenv("client_x509_cert_url"),
        "universe_domain": os.getenv("universe_domain")
    }

    logger.info("Connecting to BigQuery...")
    client = Client(credentials, 'literature-452020')

    start_time = datetime.now()

    logger.info("Fetching all papers from BigQuery...")
    query = """
        SELECT doi, date, institution, country_of_study
        FROM `literature-452020.psychology_of_poverty_literature.papers`
        WHERE country_of_study IS NOT NULL
          AND institution IS NOT NULL
    """

    df = client.execute_query(query)
    logger.info(f"Retrieved {len(df)} papers")

    logger.info("Processing data...")

    countries_set = set()
    institutions_set = set()
    institutions_by_country = {}
    papers_by_country = {}

    for _, row in df.iterrows():
        country_list = [c.strip() for c in str(row['country_of_study']).split(',')
                       if c.strip() and c.strip().lower() != 'nan']
        countries_set.update(country_list)

        try:
            inst_list = ast.literal_eval(str(row['institution']))
            if isinstance(inst_list, list):
                inst_list = [str(i) for i in inst_list if i]
            else:
                inst_list = []
        except:
            inst_list = [i.strip() for i in str(row['institution']).split(',') if i.strip()]

        institutions_set.update(inst_list)

        for country in country_list:
            if country not in institutions_by_country:
                institutions_by_country[country] = set()
                papers_by_country[country] = []
            institutions_by_country[country].update(inst_list)
            papers_by_country[country].append({
                'date': row['date'],
                'institutions': inst_list
            })

    countries = sorted(list(countries_set))
    institutions = sorted(list(institutions_set))
    institutions_by_country = {k: sorted(list(v)) for k, v in institutions_by_country.items()}

    logger.info(f"Found {len(countries)} countries, {len(institutions)} institutions")

    statistics = {}

    logger.info("Computing global statistics...")
    all_institutions = []
    for _, row in df.iterrows():
        try:
            inst_list = ast.literal_eval(str(row['institution']))
            if isinstance(inst_list, list):
                all_institutions.extend([str(i) for i in inst_list if i])
        except:
            inst_list = [i.strip() for i in str(row['institution']).split(',')]
            all_institutions.extend([i for i in inst_list if i])

    institution_counts = Counter(all_institutions)

    statistics["All__All"] = {
        'total_papers': len(df),
        'date_range': f"{df['date'].min()} – {df['date'].max()}",
        'min_date': str(df['date'].min()),
        'max_date': str(df['date'].max()),
        'countries_count': len(countries),
        'institutions_count': len(institutions),
        'top_institutions': [
            {'institution': inst, 'count': count}
            for inst, count in institution_counts.most_common(10)
        ]
    }

    logger.info(f"Computing per-country statistics...")
    for country in countries:
        country_papers = papers_by_country.get(country, [])
        if not country_papers:
            continue

        dates = [p['date'] for p in country_papers]
        country_institutions = []
        for p in country_papers:
            country_institutions.extend(p['institutions'])

        inst_counts = Counter(country_institutions)

        statistics[f"{country}__All"] = {
            'total_papers': len(country_papers),
            'date_range': f"{min(dates)} – {max(dates)}",
            'min_date': str(min(dates)),
            'max_date': str(max(dates)),
            'countries_count': 1,
            'institutions_count': len(set(country_institutions)),
            'top_institutions': [
                {'institution': inst, 'count': count}
                for inst, count in inst_counts.most_common(10)
            ]
        }

    duration = (datetime.now() - start_time).total_seconds()

    cache_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat() + 'Z',
            'schema_version': '1.0',
            'total_papers': len(df),
            'generation_duration_seconds': round(duration, 2)
        },
        'filters': {
            'countries': countries,
            'institutions': institutions,
            'institutions_by_country': institutions_by_country
        },
        'statistics': statistics
    }
    cache_path = "data/cache/statistics_cache.json"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    file_size = os.path.getsize(cache_path) / (1024 * 1024)
    logger.info(f"Statistics cache saved: {cache_path} ({file_size:.2f} MB)")

    # Generate UMAP cache
    logger.info("Fetching UMAP data from BigQuery...")
    umap_query = """
        SELECT title, doi, UMAP1, UMAP2, date
        FROM `literature-452020.psychology_of_poverty_literature.papers`
        WHERE UMAP1 IS NOT NULL
          AND UMAP2 IS NOT NULL
    """
    umap_df = client.execute_query(umap_query)
    logger.info(f"Retrieved {len(umap_df)} papers with UMAP coordinates")
    umap_cache_path = "data/cache/umap_base_data.parquet"
    umap_df.to_parquet(umap_cache_path, index=False, compression='snappy')
    umap_file_size = os.path.getsize(umap_cache_path) / (1024 * 1024)
    logger.info(f"UMAP cache saved: {umap_cache_path} ({umap_file_size:.2f} MB)")


    # generate sankey cache (from aggregated table)
    sankey_query = """
        SELECT poverty_context, study_type, mechanism, behavior, paper_count as count
        FROM `literature-452020.psychology_of_poverty_literature.fctSankeyAggregated`
    """
    logger.info("Fetching Sankey aggregated data from BigQuery...")
    sankey_df = client.execute_query(sankey_query)
    logger.info(f"Retrieved {len(sankey_df)} aggregated sankey records")
    sankey_cache_path = "data/cache/sankey_base_data.parquet"
    sankey_df.to_parquet(sankey_cache_path, index=False, compression='snappy')
    sankey_file_size = os.path.getsize(sankey_cache_path) / (1024 * 1024)
    logger.info(f"Sankey cache saved: {sankey_cache_path} ({sankey_file_size:.2f} MB)")

    # generate topics cache
    logger.info("Fetching Topics data from BigQuery...")
    topics_query = """
            SELECT *
            FROM `literature-452020.psychology_of_poverty_literature.topics`
    """
    topics_df = client.execute_query(topics_query)
    logger.info(f"Retrieved {len(topics_df)} topic records")
    topics_cache_path = "data/cache/topics_data.parquet"
    topics_df.to_parquet(topics_cache_path, index=False, compression='snappy')
    logger.info(f"Topics cache saved: {topics_cache_path}")


    total_duration = (datetime.now() - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("Cache Generation Complete")
    logger.info("=" * 60)
    logger.info(f"  Statistics Combinations: {len(statistics)}")
    logger.info(f"  Countries: {len(countries)}")
    logger.info(f"  Institutions: {len(institutions)}")
    logger.info(f"  UMAP Papers: {len(umap_df)}")
    logger.info(f"  Sankey Records: {len(sankey_df)}")
    logger.info(f"\n  Files Generated:")
    logger.info(f"    - {cache_path} ({file_size:.2f} MB)")
    logger.info(f"    - {umap_cache_path} ({umap_file_size:.2f} MB)")
    logger.info(f"    - {sankey_cache_path} ({sankey_file_size:.2f} MB)")
    logger.info(f"\n  Total Duration: {total_duration:.2f}s")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
