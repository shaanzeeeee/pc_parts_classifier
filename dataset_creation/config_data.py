import os

# Base directory for the raw dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Keyword mapping for our 11 classes massively expanded for dataset variance
CLASS_MAPPINGS = {
    "Graphics_Card": [
        "RTX graphics card", "GPU component", "PCIe graphics card PC", 
        "Nvidia RTX GPU inside gaming PC", "AMD Radeon graphics card teardown", 
        "GPU held by hand", "Watercooled graphics card block", "Graphics card plugged into motherboard",
        "PC building installing GPU", "Dual graphics cards SLI", "Intel Arc desktop GPU"
    ],
    "Motherboard": [
        "ATX motherboard", "AM5 socket motherboard", "LGA 1700 board PC",
        "Mini ITX motherboard top down", "Motherboard inside PC case naked", "Installing CPU into motherboard PC",
        "Gaming motherboard RGB", "Server motherboard dual socket", "Bare PCB motherboard components",
        "E-ATX large motherboard", "Motherboard I/O shield"
    ],
    "RAM_Stick": [
        "DDR4 RAM stick", "DDR5 memory module", "PC RAM stick RGB",
        "Installing RAM into PC motherboard", "Hand holding RAM stick memory", "Corsair vengeance DDR5",
        "Naked RAM stick green PCB", "GSkill Trident Z RAM memory PC", "Server ECC memory stick",
        "Filling all RAM slots motherboard", "Laptop SO-DIMM RAM stick"
    ],
    "CPU": [
        "Intel Core CPU processor", "AMD Ryzen CPU", "Desktop processor AM5 LGA1700",
        "CPU processor held in hand bottom pins", "CPU installed in motherboard socket", "Applying thermal paste to CPU",
        "Intel Core i9 processor box", "AMD Ryzen 9 processor macro", "Naked CPU die delidded",
        "Old CPU with bent pins", "Computer processor macro photography"
    ],
    "Power_Supply": [
        "ATX Power Supply Unit PC", "PSU 850w gold", "Modular power supply PC",
        "PC power supply cables plugged in", "SFX mini power supply case", "Installing PSU into computer case",
        "Power supply back panel switch", "Corsair RMx power supply", "White PC power supply",
        "Non-modular power supply ketchup mustard cables", "Power supply fan grill"
    ],
    "M2_NVMe_Drive": [
        "M.2 NVMe SSD", "PCIe gen 4 NVMe drive", "PC M.2 storage",
        "Installing M.2 SSD into motherboard PC", "NVMe SSD with heatsink", "Samsung 980 Pro NVMe PC",
        "Hand holding small NVMe drive", "M.2 SSD chip macro", "WD Black SN850X",
        "Motherboard M.2 slot empty", "Laptop M.2 NVMe slot"
    ],
    "AIO_Liquid_Cooler": [
        "AIO Liquid Cooler PC", "240mm AIO cooler", "360mm PC liquid cooling",
        "AIO pump block installed on CPU", "PC radiator with RGB fans", "NZXT Kraken AIO liquid cooler running",
        "PC watercooling AIO tubes", "Installing AIO liquid cooler radiator", "Corsair iCUE liquid cooler PC",
        "360mm AIO mounted on top of PC case", "White AIO liquid cooler build"
    ],
    "Air_Cooler": [
        "PC tower air cooler", "Noctua NH-D15 array", "CPU heatsink fan",
        "Dual tower CPU cooler installed", "Low profile CPU air cooler ITX", "Intel stock cooler spinning",
        "AMD Wraith Prism stock cooler", "BeQuiet Dark Rock Pro 4 cooler", "RGB tower air cooler PC",
        "Applying CPU cooler to socket", "Large heatpipes CPU cooler"
    ],
    "PC_Case": [
        "Mid tower PC case", "ATX computer case glass", "PC building case",
        "Empty PC case side panel off", "Mini ITX SFF PC case", "Dual chamber PC case Lian Li",
        "PC case front mesh panel", "Completed PC build inside case", "White tempered glass PC case",
        "Custom open air PC frame", "Full tower E-ATX computer case"
    ],
    "Good_Cable_Management": [
        "Good cable management PC build", "clean PC cable routing", "neat PC interior cables",
        "Custom sleeved extension cables PC", "PC back panel cable management tied down", "Zip tied cables PC routing",
        "Clean aesthetic PC build interior", "Straight cable combs PC build", "Hidden cables PC case",
        "Professional PC build cable management", "RGB strimer cables properly managed"
    ],
    "Bad_Cable_Management": [
        "Bad cable management PC", "PC cables rat nest", "messy PC wiring",
        "Spaghetti cables inside PC case", "Ketchup and mustard cables messy", "Unmanaged non-modular PSU cables in case",
        "Cables blocking PC airflow", "Cluttered PC back panel cables stuffed", "Horrible cable management computer",
        "Lazy PC builder cables dangling", "Untied messy wires in computer cabinet"
    ]
}

MAX_IMAGES_PER_CLASS = 500
