import re

import clean

# keys for the dictionaries to be shaped for the different root elements
node_fields = ['node_id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
node_tag_fields = ['node_id', 'key', 'value', 'type']

way_fields = ['way_id', 'user', 'uid', 'version', 'changeset', 'timestamp']
way_tag_fields = ['way_id', 'key', 'value', 'type']
way_node_fields = ['way_id', 'node_id', 'position']

relation_fields = ['relation_id', 'user', 'uid', 'version', 'changeset', 'timestamp']
relation_tag_fields = ['relation_id', 'key', 'value', 'type']
relation_member_fields = ['relation_id', 'type', 'node_id', 'way_id', 'role']

# regex for characters we do not want in the tag keys
problem_chars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def shape_element(elem, node_attr_fields=node_fields, way_attr_fields=way_fields, relation_attr_fields=relation_fields,
                  problem_chars=problem_chars, default_tag_type='regular'):
    """Clean and shape node, way and relation root elements to dictionaries."""
    
    node_attribs = {}
    way_attribs = {}
    relation_attribs = {}
    
    way_nodes = []
    relation_members = []
    
    tags = []  # handle secondary tags the same way for node, way and relation elements
    
    if elem.tag == 'node':
        # shape dictionary for 'node' element attributes
        for field in node_fields:
            if field == 'node_id':
                node_attribs[field] = elem.attrib['id']
            else:
                node_attribs[field] = elem.attrib[field]
    
    elif elem.tag == 'way':
        # shape dictionary for 'way' element attributes
        for field in way_fields:
            if field == 'way_id':
                way_attribs[field] = elem.attrib['id']
            else:
                way_attribs[field] = elem.attrib[field]
        
        # 'nd' children of 'way' elements
        position = 0    
        for node in elem.iter('nd'):
            node_dict = {}
            
            node_dict['way_id'] = elem.attrib['id']
            node_dict['node_id'] = node.attrib['ref']
            node_dict['position'] = position
            
            way_nodes.append(node_dict) 
            position += 1
    
    elif elem.tag == 'relation':
        # shape dictionary for 'relation' element attributes
        for field in relation_fields:
            if field == 'relation_id':
                relation_attribs[field] = elem.attrib['id']
            else:
                relation_attribs[field] = elem.attrib[field]
        
        # 'member' children of 'way' elements   
        for member in elem.iter('member'):
            member_dict = {}

            member_dict['relation_id'] = elem.attrib['id']
            member_dict['type'] = member.attrib['type']
                
            if member.attrib['type'] == 'node':
                member_dict['node_id'] = member.attrib['ref']
                member_dict['way_id'] = None
            else:
                member_dict['way_id'] = member.attrib['ref']
                member_dict['node_id'] = None
                
            member_dict['role'] = member.attrib['role']
                
            relation_members.append(member_dict)
    
    for child in elem:
        # 'tag' children of all root elements
        if child.tag == 'tag':
            if problem_chars.match(child.attrib['k']):
                print(child.attrib['k'])
                continue
            else:
                tag_dict = {}
                
                if elem.tag == 'node':
                    tag_dict['node_id'] = elem.attrib['id']
                elif elem.tag == 'way':
                    tag_dict['way_id'] = elem.attrib['id']
                elif elem.tag == 'relation':
                    tag_dict['relation_id'] = elem.attrib['id']
                
                if ':' not in child.attrib['k']:
                    tag_dict['key'] = child.attrib['k']
                    tag_dict['type'] = default_tag_type
                else:
                    loc = child.attrib['k'].find(':')
                    tag_dict['key'] = child.attrib['k'][loc + 1:]
                    tag_dict['type'] = child.attrib['k'][:loc]
                
                # cleaning of tag values
                if tag_dict['key'] == 'phone':
                    phone_number = child.attrib['v']
                    phone_number = clean.clean_phone_number(phone_number)
                    tag_dict['value'] = phone_number
                elif tag_dict['key'] == 'website':
                    website = child.attrib['v']
                    website = clean.clean_website(website)
                    tag_dict['value'] = website
                elif tag_dict['key'] == 'city':
                    city = child.attrib['v']
                    city = clean.clean_city(city)
                    tag_dict['value'] = city
                elif tag_dict['key'] == 'housenumber':
                    house_number = child.attrib['v']
                    house_number = clean.clean_house_number(house_number)
                    tag_dict['value'] = house_number
                else:
                    tag_dict['value'] = child.attrib['v']
                
            tags.append(tag_dict)

    if elem.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif elem.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}
    elif elem.tag == 'relation':
        return {'relation': relation_attribs, 'relation_members': relation_members, 'relation_tags': tags}