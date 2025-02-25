#!/usr/bin/env python3
import os
import math
import json
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, mapping
from shapely.ops import split
import rasterio
from rasterio import features
import folium

# Attempt to import CuPy for GPU-accelerated array operations.
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available (using CuPy).")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found; falling back to CPU (NumPy) operations.")

#####################################
# PARAMETERS (adjust as needed)
#####################################

ANGLE_THRESHOLD = 45       # degrees for triangle formation
RASTER_RESOLUTION = 5      # feet per pixel in the rasterized analysis
BUILDING_OVERLAP_THRESHOLD = 80  # percent; triangles >80% covered by buildings are discarded
OFFSET_DISTANCE = 10       # feet to offset the trimming line away from the intersection

INPUT_FOLDER = './input'
OUTPUT_FOLDER = './output'

# Input dataset filenames
STREETS_FILE       = os.path.join(INPUT_FOLDER, 'streets.geojson')
BUILDINGS_FILE     = os.path.join(INPUT_FOLDER, 'bldgs.geojson')
SIDEWALK_FILE      = os.path.join(INPUT_FOLDER, 'sidewalk.geojson')
ROADBED_FILE       = os.path.join(INPUT_FOLDER, 'roadbed.geojson')
NEIGHBORHOODS_FILE = os.path.join(INPUT_FOLDER, 'CSC_Neighborhoods_Simple.geojson')

# Output filenames
OUTPUT_GEOJSON = os.path.join(OUTPUT_FOLDER, 'triangles.geojson')
OUTPUT_WEBMAP  = os.path.join(OUTPUT_FOLDER, 'triangles.html')

# Define the input CRS
INPUT_CRS = "EPSG:2263"

#####################################
# HELPER FUNCTIONS
#####################################

def load_data():
    """Load input datasets as GeoDataFrames."""
    streets   = gpd.read_file(STREETS_FILE)
    buildings = gpd.read_file(BUILDINGS_FILE)
    sidewalks = gpd.read_file(SIDEWALK_FILE)
    roadbeds  = gpd.read_file(ROADBED_FILE)
    return streets, buildings, sidewalks, roadbeds

def get_endpoints(geom):
    """Return the start and end coordinates of a LineString geometry."""
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        return coords[0], coords[-1]
    return None, None

def compute_angle(v1, v2):
    """Compute the angle (in degrees) between two vectors."""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_angle = max(min(dot / (norm1 * norm2), 1), -1)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def form_triangles(streets, street_names):
    """
    For each street segment endpoint, find all segments sharing that endpoint.
    For each pair, if the angle between the segments is less than the threshold,
    form a triangle by connecting the common endpoint and the other endpoints.
    Returns a list of triangle features with the original street segment IDs,
    stores the intersection (common endpoint) as 'intersection', and a "Street"
    field combining the two street names. If both segments have the same 'Street'
    value (ignoring case and whitespace), the triangle is dropped.
    """
    endpoints = {}
    for idx, row in streets.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            continue
        ep1, ep2 = get_endpoints(geom)
        for ep, other in [(ep1, ep2), (ep2, ep1)]:
            key = (round(ep[0], 3), round(ep[1], 3))
            endpoints.setdefault(key, []).append((idx, other))
    
    triangles = []
    triangle_ref = {}
    for common_ep, seg_list in endpoints.items():
        if len(seg_list) < 2:
            continue
        n = len(seg_list)
        for i in range(n):
            for j in range(i+1, n):
                seg_id1, other1 = seg_list[i]
                seg_id2, other2 = seg_list[j]
                v1 = (other1[0] - common_ep[0], other1[1] - common_ep[1])
                v2 = (other2[0] - common_ep[0], other2[1] - common_ep[1])
                angle = compute_angle(v1, v2)
                if angle < ANGLE_THRESHOLD:
                    poly = Polygon([
                        (common_ep[0], common_ep[1]),
                        (other1[0], other1[1]),
                        (other2[0], other2[1])
                    ])
                    if poly.is_valid and poly.area > 0:
                        sorted_coords = sorted([(round(pt[0], 3), round(pt[1], 3))
                                                for pt in list(poly.exterior.coords)[:-1]])
                        key = tuple(sorted_coords)
                        if key not in triangle_ref:
                            # Combine street names from the two segments.
                            name1 = street_names.get(seg_id1, "").strip().lower()
                            name2 = street_names.get(seg_id2, "").strip().lower()
                            # If the two street names match, drop this triangle.
                            if name1 == name2 and name1 != "":
                                continue
                            street_field = f"{street_names.get(seg_id1, '')} / {street_names.get(seg_id2, '')}"
                            triangle_ref[key] = (seg_id1, seg_id2)
                            triangles.append({
                                'geometry': poly,
                                'seg_ids': (seg_id1, seg_id2),
                                'intersection': common_ep,
                                'Street': street_field
                            })
    return triangles

def rasterize_geometry(geom, bounds, resolution):
    """Rasterize a single geometry within provided bounds."""
    minx, miny, maxx, maxy = bounds
    width  = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
    raster = features.rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    return raster, transform

def calculate_overlap(triangle, surfaces, resolution):
    """
    Rasterizes the triangle and each surface type on the same grid.
    Returns a dictionary with overlap percentages (in percent) for each surface type.
    """
    bounds = triangle.bounds
    tri_raster, _ = rasterize_geometry(triangle, bounds, resolution)
    total_pixels = np.sum(tri_raster)
    if total_pixels == 0:
        return None
    overlaps = {}
    for key, gdf in surfaces.items():
        clipped = gdf[gdf.intersects(triangle)]
        if clipped.empty:
            overlaps[key] = 0
            continue
        union_geom = clipped.geometry.union_all()
        surf_raster, _ = rasterize_geometry(union_geom, bounds, resolution)
        if GPU_AVAILABLE:
            tri_gpu = cp.asarray(tri_raster)
            surf_gpu = cp.asarray(surf_raster)
            intersection = cp.logical_and(tri_gpu, surf_gpu)
            overlap_pixels = int(cp.sum(intersection).get())
        else:
            intersection = np.logical_and(tri_raster, surf_raster)
            overlap_pixels = np.sum(intersection)
        overlaps[key] = (overlap_pixels / total_pixels) * 100
    return overlaps

def trim_triangle(triangle, intersection, building_geoms, offset_distance):
    """
    Trims the triangle:
    - Among the building polygons intersecting the triangle, identify the vertex closest to the intersection.
    - Compute a point offset by offset_distance (ft) from the intersection along the vector toward that vertex.
    - Create a line through that offset point perpendicular to the vector.
    - Split the triangle by that line and return the piece that contains the intersection.
    """
    inter_pt = Point(intersection)
    min_dist = float('inf')
    closest_vertex = None
    for geom in building_geoms.geometry:
        if geom.is_empty:
            continue
        for coord in list(geom.exterior.coords):
            pt = Point(coord)
            d = inter_pt.distance(pt)
            if d < min_dist:
                min_dist = d
                closest_vertex = pt
    if closest_vertex is None:
        return triangle

    dx = closest_vertex.x - inter_pt.x
    dy = closest_vertex.y - inter_pt.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        return triangle
    u = (dx / norm, dy / norm)
    offset_point = Point(closest_vertex.x + u[0]*offset_distance,
                         closest_vertex.y + u[1]*offset_distance)
    perp = (-u[1], u[0])
    L = 1000  # ensure the line spans the triangle
    p1 = Point(offset_point.x + perp[0]*L, offset_point.y + perp[1]*L)
    p2 = Point(offset_point.x - perp[0]*L, offset_point.y - perp[1]*L)
    trim_line = LineString([p1, p2])
    try:
        split_result = split(triangle, trim_line)
        pieces = list(split_result.geoms)
    except Exception as e:
        print("Error splitting triangle:", e)
        return triangle
    for poly in pieces:
        if poly.contains(inter_pt) or poly.touches(inter_pt):
            return poly
    return triangle

#####################################
# MAIN PROCESSING
#####################################

def main():
    start_time = time.time()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    print("Loading input data...")
    streets, buildings, sidewalks, roadbeds = load_data()

    # Build a dictionary mapping street segment index to its Street name.
    # (Assuming the streets GeoDataFrame has a field called "Street".)
    street_names = streets.set_index(streets.index)['Street'].to_dict()

    # Load neighborhoods and subset datasets.
    neighborhoods = gpd.read_file(NEIGHBORHOODS_FILE)
    if neighborhoods.crs != INPUT_CRS:
        neighborhoods = neighborhoods.to_crs(INPUT_CRS)
    union_neighborhood = neighborhoods.geometry.union_all()

    streets   = streets[streets.intersects(union_neighborhood)]
    buildings = buildings[buildings.intersects(union_neighborhood)]
    sidewalks = sidewalks[sidewalks.intersects(union_neighborhood)]
    roadbeds  = roadbeds[roadbeds.intersects(union_neighborhood)]

    print("Identifying and forming triangles from street segments...")
    triangles_data = form_triangles(streets, street_names)
    print(f"Identified {len(triangles_data)} potential triangles.")

    surfaces = {
        'building': buildings,
        'sidewalk': sidewalks,
        'roadbed': roadbeds
    }

    processed_features = []
    for tri in triangles_data:
        poly = tri['geometry']
        overlaps = calculate_overlap(poly, surfaces, RASTER_RESOLUTION)
        if overlaps is None:
            continue
        # Omit triangles that are 100% roadbed.
        if overlaps.get('roadbed', 0) >= 100:
            continue
        # If there is any building overlap, trim the triangle.
        if overlaps.get('building', 0) > 0:
            building_subset = buildings[buildings.intersects(poly)]
            trimmed_poly = trim_triangle(poly, tri['intersection'], building_subset, OFFSET_DISTANCE)
            new_overlaps = calculate_overlap(trimmed_poly, surfaces, RASTER_RESOLUTION)
            if new_overlaps is None:
                continue
            poly = trimmed_poly
            overlaps = new_overlaps

        other_pct = 100 - (overlaps.get('building', 0) +
                           overlaps.get('sidewalk', 0) +
                           overlaps.get('roadbed', 0))
        feature = {
            'type': 'Feature',
            'geometry': mapping(poly),
            'properties': {
                'area_sqft': poly.area,
                'roadbed_overlap_pct': overlaps.get('roadbed', 0),
                'sidewalk_overlap_pct': overlaps.get('sidewalk', 0),
                'building_overlap_pct': overlaps.get('building', 0),
                'other_overlap_pct': other_pct,
                'seg_ids': tri['seg_ids'],
                'Street': tri.get('Street', "")
            }
        }
        processed_features.append(feature)

    gdf_triangles = gpd.GeoDataFrame.from_features(processed_features, crs=INPUT_CRS)
    gdf_triangles_4326 = gdf_triangles.to_crs(epsg=4326)
    gdf_triangles_4326.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
    print(f"GeoJSON output written to {OUTPUT_GEOJSON}")

    neighborhoods_4326 = neighborhoods.to_crs(epsg=4326)

    if not gdf_triangles_4326.empty:
        m = folium.Map(location=[40, -73], zoom_start=10, control_scale=True,
                       tiles="CartoDB positron", name="Positron")
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr='Esri World Imagery',
            name='ESRI World Imagery',
            overlay=True,
            control=True,
            maxZoom=22
        ).add_to(m)

        def style_function(feature):
            return {
                "fillOpacity": 0,
                "color": "blue",
                "weight": 2
            }
            
        if neighborhoods_4326 is not None and not neighborhoods_4326.empty:
            folium.GeoJson(
                neighborhoods_4326,
                name="CSC_Neighborhoods",
                style_function=lambda x: {
                    "color": "black",
                    "weight": 2,
                    "fillOpacity": 0
                }
            ).add_to(m)

        triangles_json = json.loads(gdf_triangles_4326.to_json())
        folium.GeoJson(
            triangles_json,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['area_sqft', 'roadbed_overlap_pct', 'sidewalk_overlap_pct',
                        'building_overlap_pct', 'other_overlap_pct', 'Street'],
                aliases=['Area (sq ft):', 'Roadbed Overlap (%):', 'Sidewalk Overlap (%):',
                         'Building Overlap (%):', 'Other Overlap (%):', 'Street Names:'],
                sticky=True,
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['area_sqft', 'roadbed_overlap_pct', 'sidewalk_overlap_pct',
                        'building_overlap_pct', 'other_overlap_pct', 'Street'],
                aliases=['Area (sq ft):', 'Roadbed Overlap (%):', 'Sidewalk Overlap (%):',
                         'Building Overlap (%):', 'Other Overlap (%):', 'Street Names:'],
                localize=True
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)
        bounds = gdf_triangles_4326.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        m.save(OUTPUT_WEBMAP)
        print(f"Interactive webmap saved to {OUTPUT_WEBMAP}")
    else:
        print("No valid triangles found for webmap display.")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == '__main__':
    main()