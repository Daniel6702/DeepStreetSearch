from pathlib import Path
import pandas as pd
import reverse_geocoder as rg
import pycountry
import pycountry_convert as pc
import country_converter as coco

def cc_to_country_name(cc_code):
    try:
        return pycountry.countries.get(alpha_2=cc_code).name
    except AttributeError:
        return "Unknown"

def cc_to_continent(cc_code):
    try:
        continent_code = pc.country_alpha2_to_continent_code(cc_code)
        continent_map = {
            "AF": "Africa", "AS": "Asia", "EU": "Europe",
            "NA": "North America", "SA": "South America",
            "OC": "Oceania", "AN": "Antarctica",
        }
        return continent_map.get(continent_code, "Unknown")
    except KeyError:
        return "Unknown"

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    csv_path = "/home/austen/Street-View-Harvester/datasets/world/metadata_full.csv"

    df = pd.read_csv(
        csv_path,
        usecols=["panoid", "pano_lat", "pano_lon", "image_path"]
    )

    df = df.dropna(subset=["pano_lat", "pano_lon"])
    df = df[df["pano_lat"].between(-90, 90) & df["pano_lon"].between(-180, 180)]

    coords = list(zip(df["pano_lat"], df["pano_lon"]))
    results = rg.search(coords)

    geo_df = pd.DataFrame(results)
    df["admin1"] = geo_df["admin1"]
    #df["country_code"] = geo_df["cc"]

    # Country name and continent (cached lookups for speed)
    unique_ccs = geo_df["cc"].unique()
    name_map = {c: cc_to_country_name(c) for c in unique_ccs}
    continent_map = {c: cc_to_continent(c) for c in unique_ccs}
    df["country"] = geo_df["cc"].map(name_map)

    # Sub-region, e.g. "Northern Europe", "Western Asia" (UN M49 geoscheme)
    converter = coco.CountryConverter()
    subregion_map = {c: converter.convert(names=c, src="ISO2", to="UNregion") for c in unique_ccs}
    df["subregion"] = geo_df["cc"].map(subregion_map)

    df["continent"] = geo_df["cc"].map(continent_map)



    # Drop raw coordinates now that geocoding is done
    df = df.drop(columns=["pano_lat", "pano_lon"])
    # Uncomment only if you're sure you don't need to link back to images:
    df = df.drop(columns=["image_path"])

    df.to_csv("/home/austen/Street-View-Harvester/datasets/world/metadata_geocoded.csv")
    print(f"Done. Processed {len(df)} rows.")

if __name__ == "__main__":
    main()