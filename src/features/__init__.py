"""Feature engineering modules.

Each submodule builds one family of features keyed by subzone:
    demographic    elderly share, density, total population
    food_access    nearest hawker / supermarket / market and counts
    transit        bus stop and MRT exit counts within walking buffers
    accessibility  barrier-free building density and senior centre context
    landuse        residential land area share from the Master Plan
    binary         binary amenity matrix used by Apriori
    assemble       join everything into a master feature DataFrame
"""
