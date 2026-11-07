# Agent benchmark (Ollama model qwen2.5:7b-instruct)

3 generated sales tables x 14 questions = 42 questions per variant. Ground truth is computed with plain pandas on the clean table before it is made messy.

| Variant | Correct | Wrong answer | No answer | Plain-answer fallback | Needed repair | Seconds/question |
|---|---:|---:|---:|---:|---:|---:|
| raw | 7% (3/42) | 36% | 57% | 57% | 79% | 4.28 |
| autoeda | 40% (17/42) | 36% | 24% | 24% | 50% | 3.76 |

## Correct answers per question

| Question | raw | autoeda |
|---|---:|---:|
| total_sales | 0/3 | 3/3 |
| sales_by_region | 0/3 | 2/3 |
| top_categories_avg | 0/3 | 0/3 |
| order_count | 0/3 | 3/3 |
| unique_customers | 0/3 | 1/3 |
| h1_2024_sales | 0/3 | 3/3 |
| monthly_trend | 0/3 | 0/3 |
| north_sales | 0/3 | 2/3 |
| average_age | 0/3 | 0/3 |
| top_products | 0/3 | 0/3 |
| median_price | 3/3 | 3/3 |
| returned_orders | 0/3 | 0/3 |
| avg_by_category | 0/3 | 0/3 |
| quantity_by_category | 0/3 | 0/3 |
