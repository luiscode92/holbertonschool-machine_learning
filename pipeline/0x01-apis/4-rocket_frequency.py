#!/usr/bin/env python3
"""
(unofficial) SpaceX API.
Script that displays the number of launches per rocket.
Your code should not be executed when the file is imported.
"""
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    all_launches = requests.get(url).json()
    rocket_dict = {}

    for launch in all_launches:
        rocket_id = launch["rocket"]
        rocket_url = "https://api.spacexdata.com/v4/rockets/{}".\
            format(rocket_id)

        rocket = requests.get(rocket_url).json()["name"]

        if rocket in rocket_dict.keys():
            rocket_dict[rocket] += 1
        else:
            rocket_dict[rocket] = 1

    rockets = sorted(rocket_dict.items(), key=lambda kv: kv[0])
    # [('Falcon 1', 5), ('Falcon 9', 136), ('Falcon Heavy', 4)]
    rockets = sorted(rockets, key=lambda kv: kv[1], reverse=True)
    # [('Falcon 9', 136), ('Falcon 1', 5), ('Falcon Heavy', 4)]

    for rocket in rockets:
        print("{}: {}".format(rocket[0], rocket[1]))
