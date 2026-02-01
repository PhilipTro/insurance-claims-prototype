import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
PROJECT_ROOT = Path.home() / "Desktop" / "Insurance_Project" / "prototype"

np.random.seed(42)

n_samples = 200  # number of synthetic rows

# Helper functions
def random_date(start_year=2000, end_year=2015):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).date()

def random_choice(choices):
    return np.random.choice(choices)

# Generate synthetic data
df = pd.DataFrame({
    "months_as_customer": np.random.randint(1, 400, n_samples),
    "age": np.random.randint(18, 80, n_samples),
    "policy_number": np.random.randint(100000, 999999, n_samples),
    "policy_bind_date": [random_date() for _ in range(n_samples)],
    "policy_state": [random_choice(["OH","IN","IL","NY","CA","SC","TX"]) for _ in range(n_samples)],
    "policy_csl": [random_choice(["100/300","250/500","500/1000"]) for _ in range(n_samples)],
    "policy_deductable": np.random.choice([500, 1000, 2000], n_samples),
    "policy_annual_premium": np.round(np.random.uniform(800, 2000, n_samples), 2),
    "umbrella_limit": np.random.choice([0, 5000000, 6000000], n_samples),
    "insured_zip": np.random.randint(400000, 700000, n_samples),
    "insured_sex": [random_choice(["MALE","FEMALE"]) for _ in range(n_samples)],
    "insured_education_level": [random_choice(["MD","PhD","Associate","High School"]) for _ in range(n_samples)],
    "insured_occupation": [random_choice(["craft-repair","sales","tech-support","armed-forces","machine-op-inspct"]) for _ in range(n_samples)],
    "insured_hobbies": [random_choice(["reading","board-games","sleeping","bungie-jumping","sleeping"]) for _ in range(n_samples)],
    "insured_relationship": [random_choice(["husband","own-child","unmarried","other-relative"]) for _ in range(n_samples)],
    "capital-gains": np.random.randint(0, 100000, n_samples),
    "capital-loss": np.random.randint(-60000, 0, n_samples),
    "incident_date": [random_date() for _ in range(n_samples)],
    "incident_type": [random_choice(["Single Vehicle Collision","Multi-vehicle Collision","Vehicle Theft"]) for _ in range(n_samples)],
    "collision_type": [random_choice(["Side Collision","Rear Collision","Front Collision","?"]) for _ in range(n_samples)],
    "incident_severity": [random_choice(["Major Damage","Minor Damage"]) for _ in range(n_samples)],
    "authorities_contacted": [random_choice(["Police","Fire","None"]) for _ in range(n_samples)],
    "incident_state": [random_choice(["OH","IN","IL","NY","SC","VA","CA","TX"]) for _ in range(n_samples)],
    "incident_city": [random_choice(["Columbus","Arlington","Riverwood"]) for _ in range(n_samples)],
    "incident_location": [f"{np.random.randint(100,9999)} {random_choice(['4th Drive','MLK Hwy','Francis Lane','3rd Ave','Washington St'])}" for _ in range(n_samples)],
    "incident_hour_of_the_day": np.random.randint(0,24,n_samples),
    "number_of_vehicles_involved": np.random.randint(1,4,n_samples),
    "property_damage": [random_choice(["YES","NO","?"]) for _ in range(n_samples)],
    "bodily_injuries": np.random.randint(0,5,n_samples),
    "witnesses": np.random.randint(0,5,n_samples),
    "police_report_available": [random_choice(["YES","NO","?"]) for _ in range(n_samples)],
    "total_claim_amount": np.round(np.random.uniform(500,100000, n_samples),2),
    "injury_claim": np.round(np.random.uniform(0,50000, n_samples),2),
    "property_claim": np.round(np.random.uniform(0,50000, n_samples),2),
    "vehicle_claim": np.round(np.random.uniform(0,50000, n_samples),2),
    "auto_make": [random_choice(["Saab","Mercedes","Dodge","Chevrolet","Accura"]) for _ in range(n_samples)],
    "auto_model": [random_choice(["92x","E400","RAM","Tahoe","RSX"]) for _ in range(n_samples)],
    "auto_year": np.random.randint(1990,2020,n_samples),
    "fraud_reported": [random_choice(["Y","N"]) for _ in range(n_samples)],
    "_c39": [""]*n_samples  # empty column
})

# Save to CSV
df.to_csv(PROJECT_ROOT/"data"/"raw"/"sample_insurance_claims.csv", index=False)
print("Synthetic dataset generated: sample_insurance_claims.csv")
