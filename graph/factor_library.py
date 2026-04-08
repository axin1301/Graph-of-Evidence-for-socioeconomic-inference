FACTOR_LIBRARY = [
    {
        "id": "F_density_intensity",
        "factor_name": "density_intensity",
        "description": "Signals related to dense building layout, concentrated urban occupancy, and strong built-up intensity."
    },
    {
        "id": "F_commercial_activity",
        "factor_name": "commercial_activity",
        "description": "Signals related to shops, mixed-use activity, business presence, and visible urban commerce."
    },
    {
        "id": "F_residential_quality",
        "factor_name": "residential_quality",
        "description": "Signals related to housing maintenance, orderly residential streets, and perceived neighborhood quality."
    },
    {
        "id": "F_connectivity_accessibility",
        "factor_name": "connectivity_accessibility",
        "description": "Signals related to road network density, transport access, and physical connectivity."
    },
    {
        "id": "F_green_amenity",
        "factor_name": "green_amenity",
        "description": "Signals related to greenery, trees, parks, and environmental amenity."
    },
    {
        "id": "F_industrial_activity",
        "factor_name": "industrial_activity",
        "description": "Signals related to warehouses, industrial roofs, heavy infrastructure, and possible industrial land use."
    },
    {
        "id": "F_public_service_presence",
        "factor_name": "public_service_presence",
        "description": "Signals related to schools, parks, civic facilities, and public services."
    },
    {
        "id": "F_built_environment_order",
        "factor_name": "built_environment_order",
        "description": "Signals related to cleanliness, organization, paved roads, coherent layout, and maintenance."
    }
]


# factor -> claim prior for a generic single-task solver
# Values are weak priors, not final decisions.
# Keys correspond to task_spec["target_field"].
FACTOR_TASK_PRIOR = {
    "density_intensity": {
        "Population": 0.90,
        "GDP": 0.45,
        "Carbon": 0.40,
        "HousePrice": 0.15,
        "BachelorRatio": 0.10,
        "BuildHeight": 0.85
    },
    "commercial_activity": {
        "Population": 0.35,
        "GDP": 0.90,
        "Carbon": 0.35,
        "HousePrice": 0.45,
        "BachelorRatio": 0.40,
        "BuildHeight": 0.12
    },
    "residential_quality": {
        "Population": 0.15,
        "GDP": 0.25,
        "Carbon": -0.20,
        "HousePrice": 0.90,
        "BachelorRatio": 0.65,
        "BuildHeight": 0.08
    },
    "connectivity_accessibility": {
        "Population": 0.55,
        "GDP": 0.60,
        "Carbon": 0.35,
        "HousePrice": 0.30,
        "BachelorRatio": 0.35,
        "BuildHeight": 0.28
    },
    "green_amenity": {
        "Population": 0.10,
        "GDP": 0.10,
        "Carbon": -0.35,
        "HousePrice": 0.60,
        "BachelorRatio": 0.45,
        "BuildHeight": -0.05
    },
    "industrial_activity": {
        "Population": 0.10,
        "GDP": 0.35,
        "Carbon": 0.90,
        "HousePrice": -0.45,
        "BachelorRatio": -0.35,
        "BuildHeight": -0.18
    },
    "public_service_presence": {
        "Population": 0.45,
        "GDP": 0.20,
        "Carbon": 0.05,
        "HousePrice": 0.35,
        "BachelorRatio": 0.30,
        "BuildHeight": 0.08
    },
    "built_environment_order": {
        "Population": 0.15,
        "GDP": 0.20,
        "Carbon": -0.10,
        "HousePrice": 0.55,
        "BachelorRatio": 0.55,
        "BuildHeight": 0.45
    }
}


# lightweight lexical anchors for training-free evidence -> factor mapping
FACTOR_KEYWORDS = {
    "density_intensity": [
        "high building density", "dense", "density", "built-up", "closely packed",
        "mixed residential and commercial", "urban intensity", "concentrated structures",
        "moderate density of buildings", "urban and residential buildings", "apartment blocks",
        "residential and commercial buildings", "built environment", "high-rise",
        "tall buildings", "mid-rise", "multi-story", "tower blocks", "vertical skyline",
        "street enclosure", "dense mid-rise"
    ],
    "commercial_activity": [
        "commercial", "retail", "shops", "mixed-use", "business", "storefront",
        "commerce", "activity", "market", "commercial buildings", "business court",
        "commercial presence", "mixed residential and commercial buildings",
        "visible business", "active frontage", "cafes", "restaurants",
        "service-rich street", "walkable commerce", "amenity-rich"
    ],
    "residential_quality": [
        "well-maintained", "houses", "residential", "quiet streets", "organized",
        "orderly", "neighborhood quality", "clean residential", "brick houses",
        "residential street", "quiet residential area", "pleasant neighborhood",
        "high quality neighborhood", "well-kept facades", "desirable neighborhood"
    ],
    "connectivity_accessibility": [
        "road network", "roads", "connectivity", "accessible", "transport",
        "intersection", "infrastructure", "well-connected", "well-maintained road",
        "clear road", "traffic lights", "paved road", "network of roads",
        "walkable street", "pedestrian access", "transit access", "urban corridor",
        "broad avenue", "arterial road"
    ],
    "green_amenity": [
        "greenery", "trees", "parks", "green spaces", "vegetation",
        "environmental amenity", "green", "leafy trees", "trees lining the street",
        "pleasant greenery", "street trees", "landscaped"
    ],
    "industrial_activity": [
        "industrial", "warehouse", "factory", "heavy infrastructure",
        "industrial roofs", "logistics", "yard", "low-rise sheds", "warehouse blocks"
    ],
    "public_service_presence": [
        "school", "schools", "park", "parks", "public service",
        "civic", "facility", "playground", "library", "campus", "community facility"
    ],
    "built_environment_order": [
        "clean", "organized", "paved roads", "well-developed spatial organization",
        "maintained", "order", "coherent layout", "well-maintained road",
        "clean and well-maintained road", "good condition", "orderly streetscape",
        "tidy street", "well-kept environment", "high-amenity environment",
        "continuous frontage", "large building blocks", "uniform street wall"
    ]
}



