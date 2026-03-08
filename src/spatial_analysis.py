"""Spatial analysis utilities for Berlin daycare center locations.

Provides isochrone generation with the OpenRouteService API and extraction of
green and water areas from OpenStreetMap data for the KitaMap Berlin project.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import openrouteservice as ors
import osmium
import shapely.wkb as wkblib
from shapely.geometry import shape
from tqdm import tqdm

from config import (
    DAYCARE_PROCESSED_FILE,
    EXPORT_CRS,
    GREEN_AREAS_FILE,
    ISOCHRONE_RANGE_M,
    ISOCHRONES_CLEAN_FILE,
    ISOCHRONES_FILE,
    MAX_API_REQUESTS,
    OSM_PBF_FILE,
    RESULTS_DIR,
    WATER_AREAS_FILE,
)

wkb_factory = osmium.geom.WKBFactory()


class OSMAreaExtractor(osmium.SimpleHandler):
    """Extract green and water areas from OpenStreetMap data."""

    def __init__(self) -> None:
        super().__init__()
        self.green_areas = []
        self.water_areas = []
        self.progress = tqdm(desc="Extracting OSM areas", unit=" elements")

    def _extract_geometry(self, element):
        """Extract geometry from an OSM element."""
        try:
            if isinstance(element, osmium.osm.Area):
                return wkblib.loads(wkb_factory.create_multipolygon(element))
            return wkblib.loads(wkb_factory.create_linestring(element))
        except Exception:
            return None

    def _is_water_feature(self, tags):
        """Return True when an OSM element represents a water feature."""
        return (
            tags.get("natural") == "water"
            or "waterway" in tags
            or tags.get("landuse") in ["reservoir", "basin"]
            or tags.get("water") in ["lake", "river", "pond"]
        )

    def _is_green_feature(self, tags):
        """Return True when an OSM element represents a green feature."""
        return (
            tags.get("landuse")
            in [
                "grass",
                "meadow",
                "forest",
                "greenfield",
                "cemetery",
                "recreation_ground",
            ]
            or tags.get("leisure")
            in [
                "park",
                "garden",
                "playground",
                "sports_centre",
                "pitch",
                "golf_course",
            ]
            or tags.get("natural") == "wood"
            or tags.get("amenity") == "grave_yard"
        )

    def _process_element(self, element) -> None:
        """Process a single OSM element."""
        self.progress.update(1)

        geometry = self._extract_geometry(element)
        if geometry is None:
            return

        tags = dict(element.tags)
        if self._is_water_feature(tags):
            self.water_areas.append(geometry)
        elif self._is_green_feature(tags):
            self.green_areas.append(geometry)

    def area(self, area) -> None:
        """Handle OSM area elements."""
        self._process_element(area)

    def way(self, way) -> None:
        """Handle OSM way elements."""
        self._process_element(way)

    def close(self) -> None:
        """Close the extraction progress bar and print a short summary."""
        self.progress.close()
        print(f"Green areas extracted: {len(self.green_areas)}")
        print(f"Water areas extracted: {len(self.water_areas)}")


class IsochroneGenerator:
    """Generate daycare isochrones with the OpenRouteService API."""

    def __init__(self, api_key: str) -> None:
        """Initialize the OpenRouteService client."""
        self.client = ors.Client(key=api_key) if api_key != "-" else None

    def calculate_isochrones(
        self, daycare_file: Path, output_file: Path
    ) -> gpd.GeoDataFrame:
        """Calculate isochrones for all daycare locations."""
        if self.client is None:
            print("No API key provided - skipping isochrone calculation")
            return gpd.GeoDataFrame()

        daycare_centers = gpd.read_file(daycare_file).to_crs(EXPORT_CRS)
        geometries, node_ids = [], []
        request_count = 0

        print(f"Calculating isochrones for {len(daycare_centers)} daycare centers...")

        with tqdm(
            total=min(len(daycare_centers), MAX_API_REQUESTS), desc="Isochrones"
        ) as pbar:
            batch_start = time.time()

            for idx, (_, daycare) in enumerate(daycare_centers.iterrows()):
                if request_count >= MAX_API_REQUESTS:
                    print(f"API limit reached ({MAX_API_REQUESTS})")
                    break

                try:
                    result = self.client.isochrones(
                        locations=[[daycare.geometry.x, daycare.geometry.y]],
                        profile="foot-walking",
                        range=[ISOCHRONE_RANGE_M],
                        attributes=["area"],
                    )

                    geometry = shape(result["features"][0]["geometry"])
                    geometries.append(geometry)
                    node_ids.append(daycare.name)
                    request_count += 1
                    pbar.update(1)

                    if idx % 11 == 10:
                        elapsed = time.time() - batch_start
                        if elapsed < 60:
                            time.sleep(60 - elapsed)
                        batch_start = time.time()

                except Exception as exc:
                    print(
                        f"Error calculating isochrone for daycare {daycare.name}: {exc}"
                    )
                    continue

        if geometries:
            isochrones = gpd.GeoDataFrame(
                {"node_id": node_ids, "geometry": geometries}, crs=EXPORT_CRS
            )
            isochrones.to_file(output_file, driver="GeoJSON")
            print(f"Saved {len(geometries)} isochrones to {output_file}")
            return isochrones

        return gpd.GeoDataFrame()


def remove_overlapping_areas(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove overlaps between isochrone geometries."""
    if len(gdf) == 0:
        return gdf

    print("Removing overlaps...")
    geometries = list(gdf.geometry)

    for i in tqdm(range(len(geometries)), desc="Removing overlaps"):
        for j in range(i + 1, len(geometries)):
            if geometries[i].intersects(geometries[j]):
                geometries[i] = geometries[i].difference(geometries[j])

    result = gdf.copy()
    result.geometry = geometries
    return result


def merge_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge all geometries into a single feature."""
    if len(gdf) == 0:
        return gdf

    unified_geometry = gdf.union_all()
    return gpd.GeoDataFrame(geometry=[unified_geometry], crs=gdf.crs)


def extract_osm_areas(osm_file: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Extract green and water areas from an OSM PBF file."""
    print(f"Extracting areas from {osm_file}...")

    extractor = OSMAreaExtractor()
    extractor.apply_file(str(osm_file))
    extractor.close()

    green_gdf = gpd.GeoDataFrame(geometry=extractor.green_areas, crs=EXPORT_CRS)
    water_gdf = gpd.GeoDataFrame(geometry=extractor.water_areas, crs=EXPORT_CRS)

    output_dir.mkdir(parents=True, exist_ok=True)

    green_path = output_dir / GREEN_AREAS_FILE.name
    water_path = output_dir / WATER_AREAS_FILE.name

    if len(green_gdf) > 0:
        green_unified = merge_geometries(green_gdf)
        green_unified.to_file(green_path, driver="GeoJSON")
        print(f"Saved green areas to {green_path}")

    if len(water_gdf) > 0:
        water_unified = merge_geometries(water_gdf)
        water_unified.to_file(water_path, driver="GeoJSON")
        print(f"Saved water areas to {water_path}")

    return green_path, water_path


def generate_isochrones(api_key: Optional[str] = None) -> Path:
    """Generate isochrones for daycare locations."""
    if api_key is None:
        api_key = os.getenv("OPENROUTESERVICE_API_KEY", "-")

    generator = IsochroneGenerator(api_key)
    output_file = ISOCHRONES_FILE

    isochrones = generator.calculate_isochrones(DAYCARE_PROCESSED_FILE, output_file)

    if len(isochrones) > 0:
        clean_isochrones = remove_overlapping_areas(isochrones)
        clean_isochrones.to_file(ISOCHRONES_CLEAN_FILE, driver="GeoJSON")
        print(f"Saved cleaned isochrones to {ISOCHRONES_CLEAN_FILE}")
        return ISOCHRONES_CLEAN_FILE

    return output_file


def run_full_analysis(api_key: Optional[str] = None) -> None:
    """Run the full spatial analysis pipeline."""
    print("Starting spatial analysis for Berlin daycare centers\n")

    print("1. Extracting green and water areas...")
    green_path, water_path = extract_osm_areas(OSM_PBF_FILE, RESULTS_DIR)

    print("\n2. Generating isochrones...")
    isochrones_path = generate_isochrones(api_key)

    print(f"\nAnalysis complete. Results saved in: {RESULTS_DIR}")
    print(f"   Green areas: {green_path.name}")
    print(f"   Water areas: {water_path.name}")
    print(f"   Isochrones: {isochrones_path.name}")


if __name__ == "__main__":
    run_full_analysis()
