import xml.etree.ElementTree as ET
import os

# Paths
input_file = "sumo_rl/nets/gangnam/gangnam_int.rou.xml"
output_file = "sumo_rl/nets/gangnam/gangnam_int_scaled.rou.xml"

# Parse XML
tree = ET.parse(input_file)
root = tree.getroot()

# Scale factor
scale = 0.5

print(f"Scaling traffic demand in {input_file} by {scale}...")

# Iterate over flows and scale 'number'
for flow in root.findall('flow'):
    original_number = int(flow.get('number'))
    new_number = int(original_number * scale)
    flow.set('number', str(new_number))
    print(f"Flow {flow.get('id')}: {original_number} -> {new_number}")

# Save new file
tree.write(output_file, encoding='UTF-8', xml_declaration=True)
print(f"Saved scaled route file to {output_file}")
