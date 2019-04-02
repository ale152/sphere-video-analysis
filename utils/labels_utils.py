import json

def convert_walking_to_json(obj):
    json_slow = [{'video': obj.meta_list[bf].split('.csv')[0] + '.mp4', 'label':'slow'}
                 for bf in obj.all_good[obj.all_speed < 0.5]]
    json_fast = [{'video': obj.meta_list[bf].split('.csv')[0] + '.mp4', 'label':'fast'}
                 for bf in obj.all_good[obj.all_speed >= 0.5]]
    json_data = json_slow + json_fast
    with open('walking_labels.json', 'w') as f:
        json.dump(json_data, f, indent=3)