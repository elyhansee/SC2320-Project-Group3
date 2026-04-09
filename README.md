# SC2320-Project-Group3
# Spatial-Behavioral Analytics: Mapping Nutritional Vulnerability and Food Accessibility in Singapore's Aging Districts

**SC2320 — Data Mining | Group Project**

This project applies the SC2320 data-mining toolkit to a concrete civic problem: identifying Singapore subzones where elderly residents face compounded nutritional-access risk, and recommending targeted policy interventions aligned with the Healthier SG and Forward SG agendas.

---

## 1. Project Overview

We treat each of Singapore's ~60 planning subzones as the unit of analysis and build a three-phase analytical pipeline:

| Phase | Question | Techniques (Course Chapter) |
|-------|----------|------------------------------|
| **Phase 1 — Digital Basket** | Which amenities co-occur, and which subzones break the pattern? | Apriori, FP-Growth, Association Rules (Ch. 02) + PathSim on a Subzone↔Amenity bipartite graph |
| **Phase 2 — RSM Segmentation** | How can we cluster subzones along Risk, Socio-demographics, Mobility? | K-Means, DBSCAN, Hierarchical clustering (Ch. 04–05) |
| **Phase 3 — Vulnerability Prediction** | Which subzones should receive the next intervention? | Decision Trees + Random Forests (Ch. 09) |


---

## 2. Repository Structure

---

## 3. Setup Instructions
### 3.1 Python environment
### 3.2 API Keys
- **data.gov.sg** — no key required, uses the v2 poll-download endpoint
- **LTA DataMall** — free API key for bus stops / MRT exits
- **OneMap SG** — free account for subzone polygon geocoding

### 3.3 Running

## 4. Datasets
| Source | Dataset | Used in |
|--------|---------|---------|
| data.gov.sg | Resident population by subzone, age, sex (2023) | 00, 01 |
| data.gov.sg | Hawker centres (geocoded) | 00, 01 |
| data.gov.sg | Supermarket licences (FairPrice, Sheng Siong, Cold Storage) | 00, 01 |
| LTA DataMall | Bus stops & MRT station exits | 00, 01 |
| HPB Healthy 365 | Kiosk locations (public listing) | 00, 01 |
| URA Master Plan | Subzone boundary polygons | 01 (map overlays) |