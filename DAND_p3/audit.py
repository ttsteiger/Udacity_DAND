import re
import xml.etree.cElementTree as ET

def is_key(elem, key):
    """Check if the passed element contains the specified key."""
    return (elem.attrib['k'] == key)
 
# audit house number
def audit_house_number(house_number, bad_house_numbers):
    """Check if house_number passed contains a lowercase letter. If so add it to the bad_house_number set."""
    if any(c.islower() for c in house_number):
        bad_house_numbers.append(house_number)
        
# audit addr:city
def audit_city(city, bad_cities):
    """Check if passed city contains an abbreviation or any form of ZH. Add tht city to the bad_cities set if any test 
    succeeds.
    """
    city_abbrev_re = re.compile('\.')
    city_zh_re = re.compile('ZH')
    
    m1 = city_abbrev_re.search(city)
    
    m2 = city_zh_re.search(city)
    
    if m1 or m2:
        bad_cities.append(city)
    
# clean phone
# +41 xx xxx xx xx
def audit_phone_number(phone_number, bad_phone_numbers):
    # check if phone number follows the standard format
    phone_number_re = re.compile('\+[0-9]{2} [0-9]{2} [0-9]{3} [0-9]{2} [0-9]{2}')
    
    m = phone_number_re.search(phone_number)
    
    if not m:
        bad_phone_numbers.append(phone_number)

# audit website
# starts with http://www.
def audit_website(website, bad_websites):
    # check if website starts with http://www.
    website_re = re.compile('^https?://www.')
    
    m = website_re.search(website)
    
    if not m:
        bad_websites.append(website)


def audit(file_name, bad_house_numbers, bad_cities, bad_phone_numbers, bad_websites):
    for event, elem in ET.iterparse(file_name, events=('end',)):
        if elem.tag in ['node', 'way', 'relation']:
            for tag in elem.iter('tag'):
                if is_key(tag, 'addr:housenumber'):
                    audit_house_number(tag.attrib['v'], bad_house_numbers)
                elif is_key(tag, 'addr:city'):
                    audit_city(tag.attrib['v'], bad_cities)
                elif is_key(tag, 'phone'):
                    audit_phone_number(tag.attrib['v'], bad_phone_numbers)
                elif is_key(tag, 'website'):
                    audit_website(tag.attrib['v'], bad_websites)