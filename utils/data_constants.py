"""
Constants and reference data for the Clinical Trials application.
This module contains lookup dictionaries, mappings, and other constant data.
"""

# Canadian provinces and territories mapping
PROVINCE_MAPPING = {
    'ON': 'Ontario',
    'QC': 'Quebec',
    'BC': 'British Columbia',
    'AB': 'Alberta',
    'MB': 'Manitoba',
    'SK': 'Saskatchewan',
    'NS': 'Nova Scotia',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'PE': 'Prince Edward Island',
    'NT': 'Northwest Territories',
    'YT': 'Yukon',
    'NU': 'Nunavut',
    # Common city to province mappings for data cleanup
    'toronto': 'Ontario',
    'ottawa': 'Ontario',
    'mississauga': 'Ontario',
    'hamilton': 'Ontario',
    'london': 'Ontario',
    'markham': 'Ontario',
    'vaughan': 'Ontario',
    'kitchener': 'Ontario',
    'windsor': 'Ontario',
    'greater toronto area': 'Ontario',
    
    'montreal': 'Quebec',
    'quebec city': 'Quebec',
    'quebec': 'Quebec',
    'laval': 'Quebec',
    'gatineau': 'Quebec',
    'sherbrooke': 'Quebec',
    
    'vancouver': 'British Columbia',
    'victoria': 'British Columbia',
    'burnaby': 'British Columbia',
    'richmond': 'British Columbia',
    'surrey': 'British Columbia',
    'kelowna': 'British Columbia',
    'metro vancouver': 'British Columbia',
    
    'calgary': 'Alberta',
    'edmonton': 'Alberta',
    'red deer': 'Alberta',
    'lethbridge': 'Alberta',
    
    'winnipeg': 'Manitoba',
    
    'saskatoon': 'Saskatchewan',
    'regina': 'Saskatchewan',
    
    'halifax': 'Nova Scotia',
    
    'saint john': 'New Brunswick',
    'st. john': 'New Brunswick',
    'moncton': 'New Brunswick',
    'fredericton': 'New Brunswick',
    
    "st. john's": 'Newfoundland and Labrador',
    
    'charlottetown': 'Prince Edward Island'
}

# Province population data (2021 census)
PROVINCE_POPULATIONS = {
    'Ontario': 14223942,
    'Quebec': 8501833,
    'British Columbia': 5000879,
    'Alberta': 4262635,
    'Manitoba': 1342153,
    'Saskatchewan': 1132505,
    'Nova Scotia': 969383,
    'New Brunswick': 775610,
    'Newfoundland and Labrador': 510550,
    'Prince Edward Island': 154331,
    'Northwest Territories': 41070,
    'Yukon': 40232,
    'Nunavut': 36858
}



# Metro area populations
METRO_POPULATIONS = {
    'metro vancouver': 2463431,  # Metro Vancouver population
    'greater toronto area': 6417516,  # GTA population
}


# Oncology classification keywords
ONCOLOGY_KEYWORDS = [
    'cancer', 'tumor', 'tumour', 'leukemia', 'lymphoma', 'oncology', 
    'sarcoma', 'carcinoma', 'melanoma', 'neoplasm', 'neoplastic', 
    'myeloma', 'adenocarcinoma', 'oncologic', 'malignant', 'metastatic',
    'metastasis', 'neuroendocrine', 'glioma', 'glioblastoma', 'neuroblastoma',
    'astrocytoma', 'ependymoma', 'craniopharyngioma', 'osteosarcoma'
]


# Population data for Canadian cities (based on recent census data)
CITY_POPULATIONS = {
    # Ontario
    'toronto': 2731571,
    'ottawa': 934243,
    'mississauga': 721599,
    'brampton': 593638,
    'hamilton': 536917,
    'london': 383822,
    'markham': 328966,
    'vaughan': 306233,
    'kitchener': 233222,
    'windsor': 217188,
    'richmond hill': 195022,
    'oakville': 193832,
    'burlington': 183314,
    'oshawa': 159458,
    'barrie': 141434,
    'st. catharines': 133113,
    'guelph': 131794,
    'cambridge': 129920,
    'whitby': 128377,
    'kingston': 117660,
    'thunder bay': 108843,
    'waterloo': 104986,
    'milton': 110128,
    'ajax': 119677,
    'newmarket': 84224,
    'peterborough': 81032,
    'sarnia': 71594,
    'north bay': 51553,
    'orillia': 31166,
    'cobourg': 19440,
    'caledon': 76581,
    'brantford': 134203,
    'east york': 118071,  # Part of Toronto
    'north york': 691600,  # Part of Toronto
    'scarborough': 632098,  # Part of Toronto
    'etobicoke': 365143,  # Part of Toronto
    'thornhill': 112719,  # Part of Vaughan/Markham
    'woodbridge': 105228,  # Part of Vaughan
    
    # Quebec
    'montreal': 1704694,
    'quebec city': 531902,
    'quebec': 531902,  # Same as quebec city
    'laval': 422993,
    'gatineau': 276245,
    'longueuil': 239700,
    'sherbrooke': 161323,
    'saguenay': 144230,
    'trois-rivières': 134413,
    'chicoutimi': 66547,  # Part of Saguenay
    'rimouski': 47006,
    'st-jérôme': 77301,
    'ste-foy': 98659,  # Part of Quebec City
    'saint-jean-sur-richelieu': 92394,
    'baie-saint-paul': 7146,
    'pointe-claire': 31380,
    'saint-eustache': 43784,
    'la malbaie': 8271,
    
    # British Columbia
    'vancouver': 2643000, # Changed from 631486 to population of Greater Vancouver Area (TODO: will need to remove other cities)
    'surrey': 518467,
    'burnaby': 232755,
    'richmond': 198309,
    'abbotsford': 141397,
    'coquitlam': 139284,
    'kelowna': 142146,
    'victoria': 85792,
    'nanaimo': 90504,
    'kamloops': 90280,
    'chilliwack': 83788,
    'new westminster': 70996,
    'west vancouver': 42473,
    'penticton': 33761,
    'whistler': 11854,
    'cranbrook': 20499,
    'fort st. james': 1598,
    'grand forks': 4049,
    'canal flats': 668,
    'chetwynd': 2503,
    'lytton': 249,
    
    # Alberta
    'calgary': 1239220,
    'edmonton': 932546,
    'red deer': 100418,
    'lethbridge': 92729,
    'st. albert': 65589,
    'medicine hat': 63271,
    'grande prairie': 63166,
    'sherwood park': 70618,  # Part of Strathcona County
    'canmore': 13992,
    'banff': 7847,
    'camrose': 18742,
    
    # Manitoba
    'winnipeg': 705244,
    
    # Saskatchewan
    'saskatoon': 273010,
    'regina': 228928,
    
    # Nova Scotia
    'halifax': 403131,
    
    # New Brunswick
    'saint john': 67575,
    'st. john': 67575,  # Same as saint john
    'moncton': 71889,
    'fredericton': 58220,
    
    # Newfoundland and Labrador
    "st. john's": 108860,
    
    # Special cases
    'windermere': 2800,  # Part of Vaughan
    'winchester': 2394,
    'alliston': 19243,  # Part of New Tecumseth
    'coburg': 19440,  # Same as Cobourg
}

# Global dictionary for common misspellings and aliases
SPELLING_CORRECTIONS = {
    # Montreal variations
    'montréal': 'montreal',
    'montr al': 'montreal',
    'montral': 'montreal',
    'montr al,': 'montreal',
    'mtl': 'montreal',
    'mont-royal': 'montreal',
    
    # Vancouver variations
    'west vancouver': 'vancouver',
    'vancouver bc': 'vancouver',
    'van': 'vancouver',
    
    # Toronto variations
    'tor': 'toronto',
    'gta': 'toronto',

    # Hamilton variations
    'hamitlon': 'hamilton',
    
    # Quebec variations
    'qc': 'quebec',
    'qc city': 'quebec city',

    # Sherbrooke variations
    'sherbrook': 'sherbrooke',

    # Edmonton variations
    'edmonton ab': 'edmonton',
    
    # Winnipeg variations
    'winnepeg': 'winnipeg',

    # Halifax variations
    'dalhousie': 'halifax',

    # St. John's variations
    "saint johns": "st john",
    "st johns": "st john",
    
    # Other variations: if the value is None, it means to ignore province names
    'ontario': None,
    'bc': None,
    'alberta': None,

    # For "quebec", if used in a province context, force the accented version for city lookup
    'quebec': 'québec'
}

# Global dictionary for city coordinates
CITY_COORDS = {
    # Ontario
    'toronto': {'lat': 43.651070, 'lon': -79.347015},
    'ottawa': {'lat': 45.424721, 'lon': -75.695000},
    'mississauga': {'lat': 43.589045, 'lon': -79.644119},
    'hamilton': {'lat': 43.255722, 'lon': -79.871101},
    'london': {'lat': 42.983612, 'lon': -81.249725},
    'kitchener': {'lat': 43.451290, 'lon': -80.492763},
    'windsor': {'lat': 42.3149, 'lon': -83.0364},
    'ajax': {'lat': 43.8509, 'lon': -79.0205},
    'markham': {'lat': 43.8561, 'lon': -79.3370},
    'waterloo': {'lat': 43.4643, 'lon': -80.5204},
    'richmond hill': {'lat': 43.8828, 'lon': -79.4403},
    'barrie': {'lat': 44.3894, 'lon': -79.6903},
    'newmarket': {'lat': 44.0582, 'lon': -79.4622},
    'oshawa': {'lat': 43.8971, 'lon': -78.8658},
    'oakville': {'lat': 43.4675, 'lon': -79.6877},
    'burlington': {'lat': 43.3255, 'lon': -79.7990},
    'niagara falls': {'lat': 43.0896, 'lon': -79.0849},
    'etobicoke': {'lat': 43.6205, 'lon': -79.5132},
    'thornhill': {'lat': 43.8161, 'lon': -79.4269},
    'brampton': {'lat': 43.7315, 'lon': -79.7624},
    'peterborough': {'lat': 44.3091, 'lon': -78.3197},
    'north bay': {'lat': 46.3091, 'lon': -79.4608},
    'guelph': {'lat': 43.5448, 'lon': -80.2482},
    'scarborough': {'lat': 43.7764, 'lon': -79.2318},
    'st. catharines': {'lat': 43.1594, 'lon': -79.2469},
    'st catharines': {'lat': 43.1594, 'lon': -79.2469},
    'thunder bay': {'lat': 48.3809, 'lon': -89.2477},
    'north york': {'lat': 43.7615, 'lon': -79.4111},
    'sarnia': {'lat': 42.9745, 'lon': -82.4066},
    'east york': {'lat': 43.6909, 'lon': -79.3352},
    'vaughan': {'lat': 43.8563, 'lon': -79.5085},
    'cobourg': {'lat': 43.9593, 'lon': -78.1677},
    'whitby': {'lat': 43.8975, 'lon': -78.9428},
    'kingston': {'lat': 44.230687, 'lon': -76.481323},
    'orillia': {'lat': 44.6087, 'lon': -79.4207},
    'woodbridge': {'lat': 43.7758, 'lon': -79.5992},
    'brantford': {'lat': 43.1394, 'lon': -80.2644},
    'caledon': {'lat': 43.8358, 'lon': -79.8661},
    'west hamilton': {'lat': 43.2555, 'lon': -79.9100},
    
    # Quebec
    'montreal': {'lat': 45.508888, 'lon': -73.561668},
    'quebec city': {'lat': 46.813878, 'lon': -71.207981},
    'quebec': {'lat': 46.813878, 'lon': -71.207981},
    'gatineau': {'lat': 45.476545, 'lon': -75.701271},
    'sherbrooke': {'lat': 45.404242, 'lon': -71.894917},
    'laval': {'lat': 45.606649, 'lon': -73.712409},
    'chicoutimi': {'lat': 48.4283, 'lon': -71.0582},
    'rimouski': {'lat': 48.4488, 'lon': -68.5236},
    'st-jérôme': {'lat': 45.7809, 'lon': -74.0036},
    'trois-rivières': {'lat': 46.3432, 'lon': -72.5430},
    'ste-foy': {'lat': 46.7805, 'lon': -71.2872},
    'saint-jean-sur-richelieu': {'lat': 45.3007, 'lon': -73.2574},
    'baie-saint-paul': {'lat': 47.4417, 'lon': -70.5042},
    'pointe-claire': {'lat': 45.4490, 'lon': -73.8166},
    'saint-eustache': {'lat': 45.5641, 'lon': -73.9035},
    'saguenay': {'lat': 48.4260, 'lon': -71.0725},
    'la malbaie': {'lat': 47.6549, 'lon': -70.1522},
    
    # British Columbia
    'vancouver': {'lat': 49.246292, 'lon': -123.116226},
    'victoria': {'lat': 48.428329, 'lon': -123.365868},
    'burnaby': {'lat': 49.248809, 'lon': -122.980507},
    'surrey': {'lat': 49.104431, 'lon': -122.801094},
    'kelowna': {'lat': 49.887952, 'lon': -119.496010},
    'new westminster': {'lat': 49.2057, 'lon': -122.9110},
    'penticton': {'lat': 49.4991, 'lon': -119.5937},
    'west vancouver': {'lat': 49.3690, 'lon': -123.1715},
    'whistler': {'lat': 50.1162, 'lon': -122.9535},
    'abbotsford': {'lat': 49.0504, 'lon': -122.3045},
    'nanaimo': {'lat': 49.1659, 'lon': -123.9401},
    'kamloops': {'lat': 50.6745, 'lon': -120.3273},
    'cranbrook': {'lat': 49.5097, 'lon': -115.7663},
    'richmond': {'lat': 49.1666, 'lon': -123.1336},
    'fort st. james': {'lat': 54.4438, 'lon': -124.2542},
    'grand forks': {'lat': 49.0313, 'lon': -118.4447},
    'canal flats': {'lat': 50.1513, 'lon': -115.8319},
    'chetwynd': {'lat': 55.6978, 'lon': -121.6298},
    'cariboo': {'lat': 52.9283, 'lon': -122.4461},
    'lytton': {'lat': 50.2337, 'lon': -121.5820},
    
    # Alberta
    'calgary': {'lat': 51.044270, 'lon': -114.062019},
    'edmonton': {'lat': 53.544388, 'lon': -113.490929},
    'sherwood park': {'lat': 53.5413, 'lon': -113.2958},
    'red deer': {'lat': 52.269, 'lon': -113.811},
    'lethbridge': {'lat': 49.6956, 'lon': -112.8451},
    'canmore': {'lat': 51.0884, 'lon': -115.3475},
    'st. albert': {'lat': 53.6322, 'lon': -113.6275},
    'grande prairie': {'lat': 55.1707, 'lon': -118.7947},
    'banff': {'lat': 51.1784, 'lon': -115.5708},
    'camrose': {'lat': 53.0216, 'lon': -112.8335},
    'medicine hat': {'lat': 50.0405, 'lon': -110.6768},
    
    # Manitoba
    'winnipeg': {'lat': 49.895077, 'lon': -97.138451},
    
    # Saskatchewan
    'saskatoon': {'lat': 52.131802, 'lon': -106.655808},
    'regina': {'lat': 50.445210, 'lon': -104.618896},
    
    # Nova Scotia
    'halifax': {'lat': 44.648618, 'lon': -63.586002},
    
    # New Brunswick
    'saint john': {'lat': 45.273220, 'lon': -66.063308},
    'st. john': {'lat': 45.273220, 'lon': -66.063308},
    'moncton': {'lat': 46.090946, 'lon': -64.790306},
    'fredericton': {'lat': 45.9636, 'lon': -66.6431},
    
    # Newfoundland and Labrador
    "st. john's": {'lat': 47.561510, 'lon': -52.712577},
    
    # Other locations/special cases
    'windermere': {'lat': 43.6631, 'lon': -79.4749},
    'winchester': {'lat': 45.0842, 'lon': -75.3500},
    'alliston': {'lat': 44.1502, 'lon': -79.8682},
    'coburg': {'lat': 43.9593, 'lon': -78.1677},
}

# Define metro area groupings
METRO_VANCOUVER_CITIES = {
    'vancouver',
    'burnaby',
    'coquitlam',
    'delta',
    'langley',  # Covers both City and District
    'langley city',
    'langley district',
    'maple ridge',
    'new westminster',
    'north vancouver',  # Covers both City and District
    'north vancouver city',
    'north vancouver district',
    'pitt meadows',
    'port coquitlam',
    'port moody',
    'richmond',
    'surrey',
    'white rock',
    'west vancouver',
    'bowen island',
    'anmore',
    'belcarra',
    # Add common spelling variations
    'west van',
    'north van',
    'poco',  # Common abbreviation for Port Coquitlam
    'new west',  # Common abbreviation for New Westminster
}

GREATER_TORONTO_AREA_CITIES = {
    # Toronto (Core)
    'toronto',
    'north york',
    'scarborough',
    'etobicoke',
    'east york',
    'york',
    
    # Peel Region
    'mississauga',
    'brampton',
    'caledon',
    
    # York Region
    'vaughan',
    'markham',
    'richmond hill',
    'aurora',
    'newmarket',
    'king',
    'whitchurch-stouffville',
    'stouffville',
    'east gwillimbury',
    'georgina',
    
    # Durham Region
    'pickering',
    'ajax',
    'whitby',
    'oshawa',
    'clarington',
    'uxbridge',
    'scugog',
    'port perry',
    'brock',
    'beaverton',
    'cannington',
    'sunderland',
    
    # Halton Region
    'oakville',
    'burlington',
    'milton',
    'halton hills',
    'georgetown',
    'acton',
    
    # Common variations and abbreviations
    'gta',
    'the 6',
    'the six',
    'tdot',
    'the dot',
    't.o.',
    'to',
}