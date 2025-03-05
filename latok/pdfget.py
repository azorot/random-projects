from bs4 import BeautifulSoup

def extract_pine_script_reference(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_text = []

    for item in soup.find_all('div', class_='tv-pine-reference-item'):
        header = item.find('h3', class_='tv-pine-reference-item__header')
        if header:
            extracted_text.append(f"\n## {header.text.strip()}")

        description = item.find('div', class_='tv-pine-reference-item__text tv-text')
        if description:
            extracted_text.append(description.text.strip())
            extracted_text.append("")  # Add an empty line after description

        for sub_header in item.find_all('div', class_='tv-pine-reference-item__sub-header'):
            extracted_text.append(f"\n### {sub_header.text.strip()}")
            content = sub_header.find_next('div', class_='tv-pine-reference-item__text tv-text')
            if content:
                if sub_header.text.strip() in ["Syntax", "Arguments", "Returns"]:
                    extracted_text.append("```")
                    lines = content.text.strip().split('\n')
                    for line in lines:
                        if ':' in line:
                            param, desc = line.split(':', 1)
                            extracted_text.append(f"- {param.strip()} : {desc.strip()}")
                        else:
                            extracted_text.append(line.strip())
                    extracted_text.append("```")
                else:
                    extracted_text.append(content.text.strip())
            extracted_text.append("")  # Add an empty line after each subsection

        see_also = item.find('div', class_='tv-pine-reference-item__see-also')
        if see_also:
            extracted_text.append("\n### See also")
            see_also_items = [f"- {link.text.strip()}" for link in see_also.find_all('a', class_='tv-tag-label')]
            extracted_text.extend(see_also_items)

        extracted_text.append("\n")  # Add an extra newline for separation between items

    return '\n'.join(extracted_text)

# Read the HTML file
with open('/home/azoroth/projects/latok/ref.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Extract the text
extracted_text = extract_pine_script_reference(html_content)

# Save the extracted text to a file
with open('pine_script_reference.txt', 'w', encoding='utf-8') as file:
    file.write(extracted_text)
