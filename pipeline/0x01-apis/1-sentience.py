#!/usr/bin/env python3
"""
Swapi API
"""
import requests


def sentientPlanets():
    """
    Return: list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url is not None:
        results = requests.get(url).json()["results"]

        for species in results:
            if "sentient" in [species["classification"],
                              species["designation"]]:

                planet_url = species["homeworld"]

                if planet_url is not None:
                    planet = requests.get(planet_url).json()["name"]
                    planets.append(planet)

        """
        Pagination helps to limit the number of results
        to help keep network traffic in check.
        """
        # There are a lot of species in different pages
        # e.g., "next": "https://swapi-api.hbtn.io/api/species/?page=2"
        url = requests.get(url).json()["next"]

    return planets
