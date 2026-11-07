"""Create examples/sample_messy.csv: a small, deliberately messy sales table.

The file is made with a fixed random seed, so everyone gets identical data.
It contains problems that later modules will learn to find:

  * "amount" is text such as "₹1,234.50", with a few junk entries
  * "age" has one impossible value (250)
  * "unit_price" has a few extreme outliers and some missing values
  * "internal_flag" is completely empty and "country" never changes
  * the last 6 rows are exact copies of earlier rows
"""

import csv
import random
from pathlib import Path

SEED = 7
ROWS = 300
OUTPUT = Path(__file__).parent / "sample_messy.csv"

FIRST_NAMES = ["Aarav", "Vivaan", "Diya", "Ananya", "Rohan", "Isha", "Kabir", "Meera", "Arjun"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Reddy", "Nair", "Iyer", "Mehta", "Khan"]
CITIES = ["Delhi", "Mumbai", "Pune", "Chennai", "Kolkata", "Bengaluru"]
NOTES = [
    "customer asked for faster delivery next time",
    "package arrived with a damaged outer box",
    "gift wrapping requested by the buyer",
    "payment was delayed by two days",
]


def make_rows(rng: random.Random) -> list[dict]:
    rows = []

    for index in range(ROWS):
        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        order_date = f"{day:02d}-{month:02d}-2024"

        amount = f"₹{rng.uniform(200, 9000):,.2f}"
        if index in (17, 90):
            amount = "unknown"
        if index == 140:
            amount = "N/A"

        age = rng.randint(18, 70)
        if index == 55:
            age = 250

        unit_price = round(rng.uniform(80, 1200), 2)
        if index in (30, 130, 220):
            unit_price = round(rng.uniform(50000, 90000), 2)
        if index % 25 == 3:
            unit_price = ""

        notes = rng.choice(NOTES) if rng.random() < 0.7 else ""

        rows.append(
            {
                "order_id": 5000 + index,
                "order_date": order_date,
                "customer_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                "age": age,
                "city": rng.choice(CITIES),
                "amount": amount,
                "quantity": rng.randint(1, 5),
                "unit_price": unit_price,
                "is_returned": rng.choice(["yes", "no", "no", "no"]),
                "notes": notes,
                "country": "India",
                "internal_flag": "",
            }
        )

    # Exact duplicates of six earlier rows.
    rows.extend(dict(rows[position]) for position in (4, 20, 44, 80, 120, 200))
    return rows


def main() -> None:
    rng = random.Random(SEED)
    rows = make_rows(rng)

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
