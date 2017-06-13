import re

# cleaning functions
def clean_house_number(house_number):
    if any(c.islower() for c in house_number):
        for i, c in enumerate(house_number):
            if c.islower() and (i == len(house_number) - 1 or house_number[i + 1] == "," or house_number[i + 1] == " "):
                house_number = "{}{}{}".format(house_number[:i], c.upper(), house_number[i + 1:])

    return house_number
    
def clean_phone_number(phone_number):
    phone_number_re = re.compile('\+[0-9]{2} [0-9]{2} [0-9]{3} [0-9]{2} [0-9]{2}')
    m = phone_number_re.search(phone_number)
    
    if not m:
        # remove '(0)', '-' and ' '
        for ch in ["-", "(0)", " "]:
            if ch in phone_number:
                phone_number = phone_number.replace(ch, "")
        
        # set correct starting sequence 
        if phone_number[:3] != "+41":
            if phone_number[:2] == "04" or phone_number[:2] == "07":
                phone_number = "+41{}".format(phone_number[1:])
        
        # set spacing
        phone_number = "{} {} {} {} {}".format(phone_number[:3], phone_number[3:5], phone_number[5:8], 
                                               phone_number[8:10], phone_number[10:])

    return phone_number


def clean_website(website):
    website_re = re.compile('^https?://www.')
    m = website_re.search(website)
    
    if not m:
        if website[:4] == "www.":
            website = "http://{}".format(website)
        elif website[:7] == "http://" and "www." not in website:
            website = "{}www.{}".format(website[:7], website[7:])
        elif website[:8] == "https://" and "www." not in website:
            website = "{}www.{}".format(website[:8], website[8:])
        else:
            website = "http://{}".format(website)
            
    return website


def clean_city(city):
    city_abbrev_re = re.compile('\.')
    city_zh_re = re.compile('ZH')
    
    m1 = city_abbrev_re.search(city)
    if m1:
        map = [["a.", "am"], ["A.", " Albis"], ["b.", "bei"]]
        for m in map:
            city = city.replace(m[0], m[1])

    m2 = city_zh_re.search(city)
    if m2:
        if "(ZH)" not in city:
            city = city.replace("ZH", "(ZH)")

    return city