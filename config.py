#LVLM SETTINGS
MODEL = "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit"
MAX_TOKENS = 256
MAX_MODEL_LENGTH = -1 #auto
TEMPERATURE = 0.1
TOP_P = 0.95
REPETITION_PENALTY = 1.0

# General visual description
PROMPT1 = "Describe this street-view image accurately and objectively for use in a visual search dataset. " \
"Focus on the most visually distinctive and searchable characteristics of the scene, including the type of area, " \
"buildings, roads, vegetation, terrain, infrastructure, weather, and any other prominent objects or features. " \
"Describe what a person could actually recognize from the image. Prefer specific visual details over vague statements. " \
"Mention colors, materials, shapes, density, condition, and spatial arrangement when useful. " \
"Do not speculate about the exact location, country, culture, or anything that cannot reasonably be inferred from the image. " \
"Do not mention that this is an image or Street View. Write one natural paragraph of approximately 80-150 words." 

# Architecture and infrastructure
PROMPT2 = "Describe this scene with particular attention to the built environment and infrastructure. " \
"Focus on features useful for visually distinguishing one place from another: building types, architectural styles, " \
"facade and roof materials, building density, road construction, lane markings, curbs, sidewalks, cycle infrastructure, " \
"utility poles and wires, street lights, signs, barriers, drainage, fences, and other street furniture. " \
"Include small but distinctive visual details when they are clearly visible. Describe how the buildings, roads, " \
"and infrastructure are arranged relative to each other. Do not guess an exact geographic location and do not invent " \
"details that are not visible. Avoid generic filler and do not mention that you are describing an image. " \
"Write one concise, natural paragraph of approximately 80-150 words."

# Landscape, environment, and scene layout
PROMPT3 = "Describe this location primarily in terms of its natural environment and overall visual character. " \
"Focus on terrain, elevation, vegetation, climate-related appearance, openness, land use, surrounding landscape, " \
"settlement density, and the spatial layout of the scene. Consider whether the area appears urban, suburban, rural, " \
"industrial, agricultural, coastal, mountainous, forested, flat, dry, tropical, temperate, or otherwise visually distinctive. " \
"Describe important relationships such as roads following hills, buildings surrounded by fields, dense vegetation beside streets, " \
"or mountains visible behind a settlement. Only describe characteristics supported by visible evidence. " \
"Do not guess the exact location or add information that cannot be seen. Write one natural paragraph of approximately 80-150 words."

# Search-style description with geographic context
PROPMT4 = "Create a natural-language description of this location suitable for someone searching for visually similar places. " \
"The known geographic context is: " \
"Describe the scene using the visible characteristics that make this particular place distinctive, " \
"such as architecture, streets, infrastructure, vegetation, terrain, density, and atmosphere. " \
"Incorporate the supplied geographic context naturally where useful, for example a residential street in Aarhus, Denmark, " \
"but do not claim that a visual feature is typical of the location unless that can reasonably be supported. " \
"The description should sound like something a user might enter into a visual Earth search engine. " \
"Do not mention Street View or the act of describing an image. Write one concise paragraph of approximately 60-130 words." 

DATASET_PATH = "test_dataset/"










UTIL_TOOL = ""

if UTIL_TOOL == "RESET SQLITE DATABASE":
    from modules.descriptions import DescriptionStore
    from pathlib import Path
    database_path = Path(DATASET_PATH)
    for path in [
        database_path,
        Path(str(database_path) + "-wal"),
        Path(str(database_path) + "-shm"),
    ]:
        if path.exists():
            path.unlink()
    with DescriptionStore(database_path):
        pass
    print(f"Reset database: {database_path}")