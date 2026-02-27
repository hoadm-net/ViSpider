# Spider Dataset Overview

## What is Spider?

Spider is a large-scale text-to-SQL benchmark dataset for semantic parsing. It evaluates the ability of models to convert natural language questions into SQL queries across diverse database schemas.

**Official Website**: https://yale-lily.github.io/spider

## Key Features

### Cross-Domain
- Databases across many different domains (academic, music, sports, business, healthcare, etc.)
- Train and test sets use completely different databases
- Models must generalize to unseen database schemas

### Complex SQL
Unlike simple datasets (e.g., WikiSQL), Spider includes:
- Multi-table joins
- Nested subqueries
- Set operations (UNION, INTERSECT, EXCEPT)
- Aggregations with GROUP BY, HAVING, ORDER BY

### Difficulty Levels
Questions are categorized into 4 difficulty levels:
- **Easy**: Simple SELECT with basic filtering
- **Medium**: Joins and/or aggregations
- **Hard**: Multiple joins, nested queries
- **Extra Hard**: Complex nested queries, set operations

## Data Structure

Each sample contains:
- `question`: Natural language question in English
- `query`: Target SQL query
- `db_id`: Database identifier
- `hardness`: Difficulty level
- Supporting files: database schemas, table contents

## Dataset Split

- **Train**: Large set of questions from Spider and companion datasets
- **Dev**: Development set (different databases from train)
- **Test**: Test set (held out for evaluation, with separate database set)

## Why Spider?

1. **Realistic**: Real-world complexity with multi-table databases
2. **Challenging**: Requires schema understanding and SQL composition
3. **Standard Benchmark**: Widely used in NLP/DB research community
4. **Well-Documented**: Clear schema definitions and annotations

## References

- Paper: "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task" (Yu et al., EMNLP 2018)
- Dataset: Available at https://yale-lily.github.io/spider
