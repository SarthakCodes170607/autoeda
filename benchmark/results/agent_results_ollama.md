# Agent benchmark (Ollama model qwen2.5:7b-instruct, planner prompt v2)

3 generated sales tables x 14 questions = 42 questions per variant. Ground truth is computed with plain pandas on the clean table before it is made messy. The v2 prompt was written after reading the failures of v1 on these same questions, so this is not a held-out result.

| Variant | Correct | Wrong answer | No answer | Plain-answer fallback | Needed repair | Seconds/question |
|---|---:|---:|---:|---:|---:|---:|
| raw | 17% (7/42) | 26% | 57% | 62% | 62% | 4.11 |
| autoeda | 83% (35/42) | 17% | 0% | 0% | 5% | 2.87 |

## Correct answers per question

| Question | raw | autoeda |
|---|---:|---:|
| total_sales | 0/3 | 3/3 |
| sales_by_region | 0/3 | 3/3 |
| top_categories_avg | 0/3 | 0/3 |
| order_count | 0/3 | 3/3 |
| unique_customers | 3/3 | 3/3 |
| h1_2024_sales | 0/3 | 3/3 |
| monthly_trend | 0/3 | 3/3 |
| north_sales | 0/3 | 0/3 |
| average_age | 1/3 | 2/3 |
| top_products | 0/3 | 3/3 |
| median_price | 3/3 | 3/3 |
| returned_orders | 0/3 | 3/3 |
| avg_by_category | 0/3 | 3/3 |
| quantity_by_category | 0/3 | 3/3 |
