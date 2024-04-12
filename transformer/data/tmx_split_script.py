import lxml.etree as ET

# Load the TMX file
tree = ET.parse('en-no.tmx')
root = tree.getroot()

# Create a namespace map
nsmap = {'xml': 'http://www.w3.org/XML/1998/namespace'}

# Open output files for writing
source_file = open('source.txt', 'w', encoding='utf-8')
target_file = open('target.txt', 'w', encoding='utf-8')

# Iterate through translation units
for tu in root.findall('.//tu'):
    source_seg = tu.find('tuv[@xml:lang="en"]/seg', nsmap).text
    target_seg = tu.find('tuv[@xml:lang="no"]/seg', nsmap).text

    if source_seg is not None and target_seg is not None:
        # Write source (English) and target (Norwegian) segments to respective files
        source_file.write(source_seg + '\n')
        target_file.write(target_seg + '\n')

# Close output files
source_file.close()
target_file.close()