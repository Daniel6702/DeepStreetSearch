DATASET_PATH = "test_dataset/"

NUM_GEO_CLASSES = 5000
NUM_EMBED_CLASSES = 768

#Panorama split settings
NUM_VIEWS = 6
CROP_SIZE = 1024
OUTPUT_SIZE = 768

#LVLM SETTINGS
MODEL = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
MAX_TOKENS = 80
MAX_MODEL_LENGTH = 2048
TEMPERATURE = 0.1
TOP_P = 0.95
REPETITION_PENALTY = 1.0

# General visual description
PROMPT1 = (
    "Describe the scene for visual place retrieval using only clearly visible features. "
    "Cover the main characteristics of the area, buildings, roads, vegetation, terrain, and infrastructure, "
    "but give particular emphasis to features or combinations of features that make this specific scene visually distinctive. "
    "Include unusual architectural details, road designs, infrastructure, vegetation, objects, materials, colors, or spatial arrangements when visible. "
    "Prefer specific discriminative details over generic observations. "
    "Do not guess the geographic location or mention the image itself. "
    "Write one dense natural description of approximately 40-45 words."
)

# Architecture and infrastructure
PROMPT2 = (
    "Describe the built environment and street infrastructure for visual place retrieval. "
    "Cover building forms and materials, roofs and facades, road surfaces and markings, curbs, sidewalks, "
    "cycle infrastructure, poles, wires, lights, signs, barriers, fences, drainage, and street furniture when relevant. "
    "Give particular emphasis to unusual or distinctive details and combinations that could help distinguish this scene from similar places. "
    "Mention spatial arrangement when useful and only include clearly visible details. "
    "Write one dense natural description of approximately 40-45 words."
)

# Landscape, environment, and scene layout
PROMPT3 = (
    "Describe the natural environment and overall spatial character of the scene for visual place retrieval. "
    "Focus on terrain, elevation, vegetation, openness, land use, settlement density, surrounding landscape, "
    "and relationships between roads, buildings, fields, water, forests, hills, or mountains. "
    "Only include characteristics supported by visible evidence. "
    "Mention specifics that uniquely and or distinctly describe this exact place."
    "Write one dense natural description of approximately 40-45 words."
    "I will provide you with the location at the end. You may mention it and or include in the describtion if it makes sense to do so." \
    "Location: "
)

# Search-style description with geographic context
PROMPT4 = (
    "Write a concise search query for finding visually similar places. "
    "Combine the supplied geographic context with the most distinctive visible characteristics, such as architecture, "
    "road design, infrastructure, vegetation, terrain, density, and scene layout. "
    "Mention specifics that uniquely and or distinctly describe this exact place."
    "Use natural wording a person might realistically enter into a geographic image search engine. "
    "Do not add unsupported visual claims. Write approximately 50-60 words. "

    "Output only the search query itself. Do not write introductory phrases such as 'Search for', 'Find', or 'Look for'. "
    "Do not use quotation marks or explain the query. "

    "A hierarchical geographic location will be provided at the end. Every supplied location component must be mentioned "
    "in the output; do not omit or replace any part of it. Incorporate the location naturally and geographically correctly. "
    "You may additionally mention a better-known nearby city, landmark, mountain range, coast, or geographic area when it "
    "makes the location easier to understand or search for. Such additions must supplement the supplied location, never "
    "replace it, and must have a genuine geographic relationship to it. "

    "Prefer precise spatial wording. Use 'in' for places contained within the supplied region and 'near' only for features "
    "that are actually nearby. For example, a scene in Provence-Alpes-Côte d'Azur may be described as being "
    "'in Provence-Alpes-Côte d'Azur, France, Western Europe, Europe, near the Alps', rather than "
    "'near Provence-Alpes-Côte d'Azur'. "

    "Location: "
)