#!/usr/bin/env python3
"""
(unofficial) SpaceX API
Script that displays the upcoming launch with these information.
Your code should not be executed when the file is imported.
"""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    all_upcoming = requests.get(url).json()

    date = float('inf')

    for i, launch in enumerate(all_upcoming):
        if date > launch["date_unix"]:
            date = launch["date_unix"]
            index = i

    launch_name = all_upcoming[index]["name"]
    date = all_upcoming[index]["date_local"]
    rocket_id = all_upcoming[index]["rocket"]
    url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_name = requests.get(url).json()["name"]
    launchpad_id = all_upcoming[index]["launchpad"]
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    launchpad = requests.get(url).json()
    lp_name = launchpad["name"]
    lp_loc = launchpad["locality"]

    # <launch name> (<date>) <rocket name>
    # - <launchpad name> (<launchpad locality>)
    print("{} ({}) {} - {} ({})".format(launch_name,
                                        date,
                                        rocket_name,
                                        lp_name,
                                        lp_loc))
