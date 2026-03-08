#!/usr/bin/env python3
"""CLI for the KitaMap Berlin spatial analysis pipeline.

Runs the full analysis or individual steps (OSM extraction, isochrone generation).

Usage:
    python run_analysis.py                    # Full analysis
    python run_analysis.py --osm-only         # OSM area extraction only
    python run_analysis.py --isochrones-only  # Isochrone generation only
"""

import argparse
import os
import sys
from pathlib import Path

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import OSM_PBF_FILE, RESULTS_DIR
from spatial_analysis import extract_osm_areas, generate_isochrones, run_full_analysis


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spatial analysis for Berlin daycare locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouteService API key (or OPENROUTESERVICE_API_KEY env var)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--osm-only",
        action="store_true",
        help="Extract OSM areas only",
    )
    group.add_argument(
        "--isochrones-only",
        action="store_true",
        help="Calculate isochrones only",
    )

    args = parser.parse_args()
    api_key = args.api_key or os.getenv("OPENROUTESERVICE_API_KEY")

    print("KitaMap Berlin - Spatial Analysis")
    print("=" * 40)

    try:
        if args.osm_only:
            print("Extracting OSM areas only...")
            extract_osm_areas(OSM_PBF_FILE, RESULTS_DIR)
        elif args.isochrones_only:
            print("Calculating isochrones only...")
            if not api_key:
                print("Warning: no API key found")
                print("Set OPENROUTESERVICE_API_KEY or use --api-key YOUR_KEY")
            generate_isochrones(api_key)
        else:
            print("Running the full analysis...")
            if not api_key:
                print("No API key found - isochrones will be skipped")
                print("Set OPENROUTESERVICE_API_KEY for the full workflow")
            run_full_analysis(api_key)

        print("\nDone.")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
