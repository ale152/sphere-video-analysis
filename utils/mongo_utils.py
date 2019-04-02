import os

import bson
import bz2


def read_room_uid(room_path, room, use_bzip):
    # Read the id of the room selected
    if room == 'kit':
        room_file = 'SPHERE_MON_VID_KITCHEN.bson'
    elif room == 'hal':
        room_file = 'SPHERE_MON_VID_HALL.bson'
    elif room == 'liv':
        room_file = 'SPHERE_MON_VID_LIVING_ROOM.bson'
    else:
        raise Exception('Room must be either "kit", "hal" or "liv", found {} instead.'.format(room))

    if use_bzip:
        room_file += '.bz2'

    room_path = os.path.join(room_path, room_file)
    fid = open_bzip(room_path, use_bzip)
    room_data = bson.decode_file_iter(fid)
    try:
        room_data = next(room_data)
        room_uid = room_data['uid']
    except StopIteration:
        room_uid = None
        print('/!\\ No room ID found!')
    fid.close()

    return room_uid


def open_bzip(file, use_bzip):
    if use_bzip:
        fid = bz2.open(file)
    else:
        fid = open(file, 'rb')
    return fid
