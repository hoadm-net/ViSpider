# Spider Dataset Overview

## Introduction

Spider is a large-scale, complex, and cross-domain semantic parsing and text-to-SQL dataset. It was introduced in the EMNLP 2018 paper and has become a widely-used benchmark for evaluating natural language interfaces to databases.

**Key Challenge**: Models must generalize to both new SQL queries AND new database schemas.

## Dataset Statistics

### Size
- **Total Questions**: 11,840
  - Train: 8,659 (7,000 from Spider + 1,659 from others)
  - Dev: 1,034
  - Test: 2,147
- **Unique SQL Queries**: 6,448
- **Databases**: 206 across 138 different domains
  - Train: 146 databases
  - Dev: 20 databases  
  - Test: 40 databases

### Complexity
- Average SQL query length: ~20.4 tokens
- Average question length: ~13.4 tokens
- Contains complex SQL with:
  - Multiple joins
  - Nested queries
  - Set operations (UNION, INTERSECT, EXCEPT)
  - Aggregations (COUNT, SUM, AVG, MIN, MAX)
  - GROUP BY, HAVING, ORDER BY clauses

## Key Characteristics

1. **Cross-Domain**: Covers 138 different domains (e.g., academic, music, sports, business, healthcare)

2. **Complex SQL**: Unlike WikiSQL (simple single-table queries), Spider includes:
   - Multi-table joins
   - Nested subqueries
   - Complex WHERE conditions
   - Multiple aggregation functions

3. **Schema Generalization**: Train and test sets have completely different databases, requiring models to understand and adapt to new schemas

4. **Human-Annotated**: Annotated by 11 Yale students with quality control

## Data Structure

### Question-SQL Pairs
Each example contains:
- `question`: Natural language question
- `query`: Corresponding SQL query
- `db_id`: Database identifier
- `question_toks`: Tokenized question
- `query_toks`: Tokenized SQL query
- `sql`: Parsed SQL structure (nested representation)

### Database Schemas
The `tables.json` file contains:
- `db_id`: Database identifier
- `table_names`: Table names in the database
- `column_names`: Column names with table references
- `column_types`: Data types (text, number, time, boolean, others)
- `primary_keys`: Primary key column indices
- `foreign_keys`: Foreign key relationships between tables

### SQLite Databases
- Each database stored as SQLite3 file
- Contains actual table content for execution-based evaluation

## Evaluation Metrics

1. **Exact Set Match**: Component-wise comparison of SQL clauses (official metric until 2020)
2. **Test Suite Accuracy**: Uses multiple test cases per query (official metric since 2020)
3. **Execution Accuracy**: Compares query execution results

## Sample Domains

The dataset includes diverse domains such as:
- academic, aircraft, apartment_rentals, baseball, battle_death
- browser_web, car, cinema, college, concert_singer
- customer_complaints, department_store, e_learning, flight, hospital
- insurance, music, restaurant, student, world_1
- And 118 more domains...
