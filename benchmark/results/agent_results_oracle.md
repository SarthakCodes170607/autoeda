# Agent benchmark (scripted oracle planner)

5 generated sales tables x 14 questions = 70 questions per variant. Ground truth is computed with plain pandas on the clean table before it is made messy.

| Variant | Correct | Wrong answer | No answer | Plain-answer fallback | Needed repair | Seconds/question |
|---|---:|---:|---:|---:|---:|---:|
| raw | 14% (10/70) | 29% | 57% | 57% | 57% | 0.43 |
| autoeda | 100% (70/70) | 0% | 0% | 0% | 0% | 0.26 |

## Correct answers per question

| Question | raw | autoeda |
|---|---:|---:|
| total_sales | 0/5 | 5/5 |
| sales_by_region | 0/5 | 5/5 |
| top_categories_avg | 0/5 | 5/5 |
| order_count | 0/5 | 5/5 |
| unique_customers | 5/5 | 5/5 |
| h1_2024_sales | 0/5 | 5/5 |
| monthly_trend | 0/5 | 5/5 |
| north_sales | 0/5 | 5/5 |
| average_age | 0/5 | 5/5 |
| top_products | 0/5 | 5/5 |
| median_price | 5/5 | 5/5 |
| returned_orders | 0/5 | 5/5 |
| avg_by_category | 0/5 | 5/5 |
| quantity_by_category | 0/5 | 5/5 |
