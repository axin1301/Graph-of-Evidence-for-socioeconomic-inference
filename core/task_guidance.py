from textwrap import dedent


_GUIDANCE = {
    "satellite": {
        "default": """
        Retrieved task guidance for satellite interpretation:
        - Treat overhead imagery as structural context: built density, road layout, land use, water adjacency, greenery, and large-scale organization.
        - Avoid turning generic urban form into strong positive socioeconomic evidence.
        - If the scene is weakly informative, mixed, or dominated by non-built surfaces, keep implication near neutral.
        """,
        "gdp": """
        Retrieved task guidance for satellite interpretation of GDP:
        - Focus on large-scale economic intensity: extensive commercial cores, transport hubs, major infrastructure, logistics concentration, and broad, consistently serviced urban fabric.
        - Dense settlement alone is not strong GDP evidence.
        - Ordinary residential grids or repetitive apartment blocks should stay near neutral unless paired with stronger economic signals.
        """,
        "population": """
        Retrieved task guidance for satellite interpretation of population:
        - Focus on settlement intensity, built-up coverage, fine-grained parcelization, and broad residential density.
        - Separate population intensity from wealth or maintenance quality.
        - Sparse or fragmented development should lower implication.
        """,
        "houseprice": """
        Retrieved task guidance for satellite interpretation of house price:
        - Use satellite imagery mainly as neighborhood context rather than direct property-value evidence.
        - Residential order, greenery, water adjacency, and access may be mildly informative, but overhead views alone should rarely imply very high house price.
        - Industrial land, logistics zones, vacant parcels, and weak neighborhood amenity context should be treated as weak or negative contextual evidence.
        """,
        "carbon": """
        Retrieved task guidance for satellite interpretation of carbon:
        - Focus on built intensity, major roads, large paved surfaces, industrial land use, logistics facilities, and transport infrastructure.
        - Greener, sparse, and weakly built environments are weaker carbon signals.
        - Dense residential areas are only moderately informative unless paired with stronger mobility or industrial intensity.
        """,
        "crime": """
        Retrieved task guidance for satellite interpretation of violent crime:
        - Use overhead imagery cautiously and treat it as broad urban context rather than direct crime evidence.
        - Dense roads, mixed land use, large parking areas, transport corridors, fragmented industrial fabric, and disorderly vacant parcels may indicate more complex or risk-prone urban environments, but should remain only moderately informative.
        - Stable residential layout, visible greenery, lower traffic intensity, and orderly neighborhood structure are weaker risk signals.
        - Do not infer crime directly from density alone.
        """,
        "violentcrime": """
        Retrieved task guidance for satellite interpretation of violent crime:
        - Use overhead imagery cautiously and treat it as broad urban context rather than direct crime evidence.
        - Dense roads, mixed land use, large parking areas, transport corridors, fragmented industrial fabric, and disorderly vacant parcels may indicate more complex or risk-prone urban environments, but should remain only moderately informative.
        - Stable residential layout, visible greenery, lower traffic intensity, and orderly neighborhood structure are weaker risk signals.
        - Do not infer crime directly from density alone.
        """,
        "bachelorratio": """
        Retrieved task guidance for satellite interpretation of bachelor ratio:
        - Focus on neighborhood context linked to educational and socioeconomic opportunity, such as orderly urban form, strong service access, mixed-use areas, campus-like structure, and high-quality residential environment.
        - Large industrial zones, logistics fabric, fragmented vacant land, or weak neighborhood amenity structure are weaker or negative contextual signals.
        - Treat overhead evidence as indirect context only; avoid strong claims unless multiple contextual cues align.
        """,
        "buildheight": """
        Retrieved task guidance for satellite interpretation of building height:
        - Focus on overhead signals associated with taller buildings: dense shadow patterns, large building footprints with compact spacing, central business districts, tower clusters, and high-intensity urban cores.
        - Sparse detached structures, low-rise industrial sheds, suburban parcels, and wide spacing are weaker or negative height signals.
        - Building height is a physical attribute, so morphology should matter more than socioeconomic quality.
        """,
        "buildingheight": """
        Retrieved task guidance for satellite interpretation of building height:
        - Focus on overhead signals associated with taller buildings: dense shadow patterns, large building footprints with compact spacing, central business districts, tower clusters, and high-intensity urban cores.
        - Sparse detached structures, low-rise industrial sheds, suburban parcels, and wide spacing are weaker or negative height signals.
        - Building height is a physical attribute, so morphology should matter more than socioeconomic quality.
        """,
    },
    "street": {
        "default": """
        Retrieved task guidance for street-view interpretation:
        - Focus on visible built-environment quality, activity intensity, accessibility, maintenance, and amenity signals.
        - Keep implication near neutral when the scene is weakly informative, ambiguous, or only partially visible.
        - Do not infer socioeconomic level from the mere presence of roads, buildings, or vehicles.
        """,
        "gdp": """
        Retrieved task guidance for street-view interpretation of GDP:
        - Focus on visible commercial intensity, service quality, mixed-use vitality, maintained public realm, mobility, and broad urban functionality.
        - Scenic waterfront, residential-only scenes, or generic apartment blocks are weak GDP evidence.
        - Clear commerce, active frontage, high-quality access, and sustained urban intensity are stronger GDP signals.
        """,
        "population": """
        Retrieved task guidance for street-view interpretation of population:
        - Focus on residential intensity, occupancy cues, frequent buildings, limited open space, and strong street enclosure.
        - Wealth or aesthetic quality does not necessarily imply high population.
        - Detached, spacious, weakly built-up scenes are more consistent with lower population intensity.
        """,
        "houseprice": """
        Retrieved task guidance for street-view interpretation of house price:
        - Focus on neighborhood desirability: building maintenance, facade quality, orderliness, cleanliness, greenery, walkability, and amenity quality.
        - Industrial, logistics, warehouse, infrastructure-heavy, bridge-underpass, vacant, or visibly deteriorated scenes are weak or negative direct evidence for house price.
        - Generic traffic or urban activity alone should not imply high house price.
        """,
        "carbon": """
        Retrieved task guidance for street-view interpretation of carbon:
        - Focus on traffic intensity, transport load, commercial or industrial activity, heavy infrastructure use, and signals of high urban activity intensity.
        - Low activity, sparse traffic, and weak built intensity should lower implication.
        - Mixed scenes should remain near neutral unless strong transport or activity cues are visible.
        """,
        "crime": """
        Retrieved task guidance for street-view interpretation of violent crime:
        - Focus on visible disorder, poor maintenance, vacancy, heavy security barriers, weak pedestrian comfort, low neighborhood care, and harsh infrastructure context as possible positive crime-risk signals.
        - Orderly, well-maintained, active, and comfortable residential or mixed-use streetscapes are weaker crime-risk signals.
        - Avoid using demographic stereotypes; rely only on visible environmental conditions.
        """,
        "violentcrime": """
        Retrieved task guidance for street-view interpretation of violent crime:
        - Focus on visible disorder, poor maintenance, vacancy, heavy security barriers, weak pedestrian comfort, low neighborhood care, and harsh infrastructure context as possible positive crime-risk signals.
        - Orderly, well-maintained, active, and comfortable residential or mixed-use streetscapes are weaker crime-risk signals.
        - Avoid using demographic stereotypes; rely only on visible environmental conditions.
        """,
        "bachelorratio": """
        Retrieved task guidance for street-view interpretation of bachelor ratio:
        - Focus on cues of educational and professional neighborhood context, such as well-maintained built environment, service richness, walkability, mixed-use vitality, campus-like or office-adjacent settings, and higher-quality amenities.
        - Industrial, heavily deteriorated, vacant, or weak-amenity scenes are weaker or negative signals.
        - Treat this as indirect contextual inference rather than a direct demographic readout.
        """,
        "buildheight": """
        Retrieved task guidance for street-view interpretation of building height:
        - Focus on directly visible physical height cues: number of floors, facade scale, verticality, tower presence, and street enclosure created by tall structures.
        - Industrial sheds, warehouses, detached houses, and low-rise blocks are negative evidence for high building height.
        - For this task, physical form matters more than perceived wealth or activity.
        """,
        "buildingheight": """
        Retrieved task guidance for street-view interpretation of building height:
        - Focus on directly visible physical height cues: number of floors, facade scale, verticality, tower presence, and street enclosure created by tall structures.
        - Industrial sheds, warehouses, detached houses, and low-rise blocks are negative evidence for high building height.
        - For this task, physical form matters more than perceived wealth or activity.
        """,
    },
}


_TASK_ALIASES = {
    "gdp": "gdp",
    "population": "population",
    "pop": "population",
    "carbon": "carbon",
    "co2": "carbon",
    "carbonemission": "carbon",
    "carbon_emission": "carbon",
    "houseprice": "houseprice",
    "house_price": "houseprice",
    "bachelorratio": "bachelorratio",
    "bachelor_ratio": "bachelorratio",
    "violentcrime": "violentcrime",
    "violent_crime": "violentcrime",
    "crime": "violentcrime",
    "buildheight": "buildheight",
    "buildingheight": "buildheight",
    "build_height": "buildheight",
}


def canonicalize_task_guidance_key(task_name: str) -> str:
    raw = str(task_name or "").strip().lower()
    if not raw:
        return "default"
    return _TASK_ALIASES.get(raw, raw)


def get_task_guidance(task_name: str, modality: str) -> str:
    task_key = canonicalize_task_guidance_key(task_name)
    modality_key = str(modality or "").strip().lower()
    modality_guidance = _GUIDANCE.get(modality_key, {})
    text = modality_guidance.get(task_key) or modality_guidance.get("default", "")
    return dedent(text).strip()


