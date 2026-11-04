"""Generate a synthetic engineering document summarization dataset for testing.

This script creates a small JSONL dataset with realistic structure.
The content is synthetic placeholder text, not real engineering documents.
For production use, replace with actual data ingestion from your source.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SAMPLE_DOCUMENTS = [
    {
        "topic": "structural load analysis",
        "source": (
            "The structural load analysis for Building 7A was conducted per ASCE 7-22 "
            "standards. Dead loads were calculated at 75 psf for the composite floor "
            "system, including a 3-inch lightweight concrete topping on 2-inch metal "
            "deck. Live loads of 50 psf were applied to office areas and 100 psf to "
            "mechanical rooms. Wind loads were computed using Exposure Category B with "
            "a basic wind speed of 115 mph. The lateral force resisting system consists "
            "of special concentrically braced steel frames in both orthogonal directions. "
            "Seismic design was governed by Site Class D conditions with an Ss of 0.45g "
            "and S1 of 0.18g, yielding a Seismic Design Category B classification. "
            "Foundation loads were transmitted to drilled shaft elements bearing on the "
            "limestone formation at approximately 35 feet below grade."
        ),
        "summary": (
            "Structural analysis for Building 7A per ASCE 7-22 found dead loads of "
            "75 psf and live loads of 50 to 100 psf. Wind loads used 115 mph basic "
            "speed under Exposure B. The lateral system uses special concentrically "
            "braced steel frames. Seismic design at SDC B. Foundations are drilled "
            "shafts to limestone at 35 feet."
        ),
    },
    {
        "topic": "HVAC system commissioning",
        "source": (
            "The HVAC commissioning report for the data center cooling system documents "
            "functional performance testing of two 500-ton centrifugal chillers, four "
            "chilled water pumps, and twelve computer room air handlers. Chiller COP was "
            "measured at 6.2 under full load conditions, exceeding the design target of "
            "5.8. Supply air temperature was maintained within plus or minus 0.5 degrees "
            "Fahrenheit of the 55 degree setpoint across all CRAHs. Variable frequency "
            "drives on the chilled water pumps demonstrated stable operation from 30 to "
            "100 percent speed. The economizer mode was verified to engage at 55 degrees "
            "Fahrenheit outdoor dry-bulb temperature, reducing compressor energy by 40 "
            "percent during winter months. All building automation system points were "
            "verified for correct trend logging and alarm setpoints."
        ),
        "summary": (
            "HVAC commissioning verified two 500-ton chillers at 6.2 COP exceeding the "
            "5.8 target. Twelve CRAHs held supply air within 0.5 degrees of setpoint. "
            "VFDs operated stably from 30 to 100 percent speed. Economizer mode at 55 "
            "degrees F reduced compressor energy by 40 percent. BAS points confirmed."
        ),
    },
    {
        "topic": "geotechnical investigation",
        "source": (
            "The geotechnical investigation for the proposed highway interchange included "
            "20 soil borings to depths of 60 to 100 feet, 10 cone penetration tests, and "
            "laboratory testing of 85 soil samples. Subsurface conditions consist of 8 to "
            "12 feet of sandy clay fill overlying native alluvial deposits of medium dense "
            "to dense sand and gravel. Groundwater was encountered at depths of 15 to 22 "
            "feet below existing grade. The recommended foundation system for bridge piers "
            "is 48-inch drilled shafts socketed a minimum of 10 feet into the underlying "
            "sandstone formation. Allowable bearing pressure for the sandstone is 30 ksf. "
            "Settlement analysis indicates less than 0.5 inches of total settlement under "
            "design loads. Liquefaction potential was evaluated and determined to be low "
            "based on SPT blow counts exceeding 25 in the granular deposits."
        ),
        "summary": (
            "Geotechnical investigation with 20 borings and 10 CPTs found sandy clay fill "
            "over alluvial sand and gravel. Groundwater at 15 to 22 feet. Recommended "
            "48-inch drilled shafts into sandstone at 30 ksf bearing. Settlement below "
            "0.5 inches. Low liquefaction risk with SPT N over 25."
        ),
    },
    {
        "topic": "electrical distribution design",
        "source": (
            "The electrical distribution design for the manufacturing facility includes a "
            "15 kV utility service entrance with two 2500 kVA pad-mounted transformers "
            "in a main-tie-main configuration for redundancy. The 480V switchgear feeds "
            "three motor control centers serving the production line equipment, with a "
            "total connected load of 3200 kW and a demand factor of 0.75. Emergency power "
            "is provided by a 1500 kW diesel generator with a 500-gallon belly tank "
            "providing 8 hours of runtime at full load. An automatic transfer switch rated "
            "at 3000A handles the switchover within 10 seconds. Power quality monitoring "
            "equipment tracks harmonic distortion from the variable frequency drives, "
            "with THD measured at 4.8 percent, below the IEEE 519 limit of 5 percent. "
            "Arc flash analysis was performed per NFPA 70E with incident energy levels "
            "documented at each panel."
        ),
        "summary": (
            "Electrical design uses two 2500 kVA transformers in main-tie-main config. "
            "480V distribution serves 3200 kW connected load at 0.75 demand factor. "
            "A 1500 kW generator provides 8 hours of backup. THD at 4.8 percent meets "
            "IEEE 519 limits. Arc flash analysis completed per NFPA 70E."
        ),
    },
    {
        "topic": "water treatment process",
        "source": (
            "The water treatment plant expansion adds 5 MGD of capacity using membrane "
            "bioreactor technology. The MBR system combines activated sludge biological "
            "treatment with ultrafiltration membrane separation, achieving effluent "
            "turbidity below 0.1 NTU and BOD5 below 2 mg per liter. The biological "
            "nutrient removal process includes an anaerobic zone for phosphorus release, "
            "an anoxic zone for denitrification, and an aerobic zone for nitrification "
            "and carbon oxidation. Total nitrogen removal achieves less than 3 mg per "
            "liter in the effluent. Membrane flux is maintained at 15 gallons per square "
            "foot per day with chemical cleaning cycles every 90 days. Energy consumption "
            "for the MBR process is approximately 1.2 kWh per cubic meter of treated water."
        ),
        "summary": (
            "Water treatment expansion adds 5 MGD using MBR technology. Effluent quality "
            "achieves turbidity below 0.1 NTU and BOD5 below 2 mg/L. BNR process removes "
            "total nitrogen to below 3 mg/L. Membrane flux at 15 GFD with 90-day cleaning. "
            "Energy use is 1.2 kWh per cubic meter."
        ),
    },
]


def generate_sample_dataset(
    output_path: str | Path = "data/sample/eng_summarization_sample.jsonl",
    num_records: int = 100,
    seed: int = 42,
) -> Path:
    """Generate a synthetic JSONL summarization dataset.

    NOTE: This dataset contains synthetic placeholder text for pipeline testing.
    Replace with actual data for production training and evaluation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    records = []

    for i in range(num_records):
        template = rng.choice(SAMPLE_DOCUMENTS)
        noise_suffix = f" Project reference: PRJ-{rng.randint(1000, 9999)}."
        split_label = (
            "train"
            if i < int(num_records * 0.8)
            else ("val" if i < int(num_records * 0.9) else "test")
        )
        record = {
            "document_id": f"DOC-{i:04d}",
            "source_text": template["source"] + noise_suffix,
            "target_summary": template["summary"],
            "split": split_label,
            "topic": template["topic"],
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {num_records} synthetic records at {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_dataset()
