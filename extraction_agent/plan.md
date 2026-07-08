Overall Architecture
                         Excel / CSV Metadata
                                 │
                                 ▼
                    Metadata Ingestion Pipeline
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
      ▼                          ▼                          ▼
 DuckDB/Postgres           Vector Database           Join Graph
(Structured Metadata)    (Semantic Search)         (NetworkX)
      │                          │                          │
      └──────────────┬───────────┴──────────────┬───────────┘
                     ▼
             Metadata Service Layer
         (Search APIs / LangGraph Tools)
                     │
                     ▼
               LangGraph Workflow
                     │
                     ▼
              Generated PySpark Code
Step 1 : Metadata Ingestion

Input

metadata.xlsx

├── Mart Sheet
├── Table Sheet
├── Column Sheet
├── Relationship Sheet

Parser

Excel

↓

Parser

↓

Normalization

↓

Metadata Objects
Step 2 : Store in DuckDB
mart_metadata

| mart_name | description | owner | refresh_frequency |
|------------|------------|----------------|

Example

| FINONE | Loan Origination Mart |

table_metadata

| mart_name | table_name | description | partition_flag | partition_column |

Example

| FINONE | LOAN_MASTER | Stores loan applications |

column_metadata ⭐

One row per column.

| mart | table | column | datatype | definition | nullable | sample | key | completeness | uniqueness |

Example

| FINONE | LOAN_MASTER | SANCTION_AMOUNT | DECIMAL | Approved Loan Amount |

relationship_metadata ⭐

This becomes your join source.

| source_table | source_column | target_table | target_column | relationship |

Example

| LOAN_MASTER | CUSTOMER_ID | CUSTOMER | CUSTOMER_ID | MANY_TO_ONE |

value_mapping ⭐

Generated manually or using LLM.

| column | business_term | actual_value |

Example

| PRODUCT_CODE | Home Loan | HL |

| PRODUCT_CODE | Personal Loan | PL |

| STATUS | Sanctioned | S |

Without this table your LLM will hallucinate filter values.

business_glossary

Generated using LLM.

| business_term | mart | table | column |

Example

| Loan Amount | FINONE | LOAN_MASTER | SANCTION_AMOUNT |

statistics_metadata

| table | column | null% | unique% | min | max |

Useful later for optimization.

Step 3 : Vector DB

Do NOT embed rows.

Embed documents.

Column Embedding

Every column becomes one document.

Example

ID

FINONE.LOAN_MASTER.SANCTION_AMOUNT

Document

Mart:
FINONE

Mart Description:
Loan servicing mart

Table:
LOAN_MASTER

Table Description:
Stores one record per loan.

Column:
SANCTION_AMOUNT

Business Meaning:
Approved loan amount after underwriting.

Datatype:
Decimal

Sample:
500000

Nullable:
No

Aliases:
Loan Amount
Approved Amount
Sanction Value
Loan Value

One embedding.

Total

≈6000 vectors
Table Embedding

One vector per table.

Example

Table

LOAN_MASTER

Purpose

Loan lifecycle table.

Contains

Loan Amount

Customer

Disbursement

EMI

Sanction Date

Relationships

CUSTOMER

PAYMENT

EMI

Total

≈350 vectors
Mart Embedding

One vector.

Example

FINONE

Contains

Loans

Applications

EMI

Collections

Disbursement

Only

8 vectors
Step 4 : Join Graph

During ingestion

Read

relationship_metadata

Build graph.

graph.add_edge(
    "LOAN_MASTER",
    "CUSTOMER",
    source_column="CUSTOMER_ID",
    target_column="CUSTOMER_ID"
)

Only once.

Runtime

NetworkX Graph

Graph

CUSTOMER

     │

     │ customer_id

     ▼

LOAN_MASTER

     │

     ▼

DISBURSEMENT

     │

     ▼

PAYMENT

     │

     ▼

BUREAU
Step 5 : LangGraph Workflow
START

↓

User Query

↓

Intent Extraction

↓

Metadata Retrieval

↓

Column Resolution

↓

Table Resolution

↓

Join Planner

↓

Business Rule Resolution

↓

Logical Query Planner

↓

PySpark Generator

↓

Metadata Validator

↓

PySpark Optimizer

↓

END
Metadata Retrieval Node

Uses

Vector Search

Searches

mart collection

table collection

column collection

Returns

Top Tables

Top Columns

Confidence
Column Resolution

Suppose user asks

Loan Amount

Vector DB

↓

SANCTION_AMOUNT

0.97

LOAN_AMOUNT

0.91

DuckDB

↓

Returns

Datatype

Definition

Table

Mart
Join Planner

Input

LOAN_MASTER

CUSTOMER

BUREAU

Graph

↓

Shortest Path

Returns

LOAN_MASTER

↓

CUSTOMER

↓

BUREAU

No LLM.

Business Rule Resolver

User

Home Loan

DuckDB

value_mapping

↓

PRODUCT_CODE='HL'

User

Sanctioned

↓

STATUS='S'
Logical Query Plan ⭐

This is the most important intermediate artifact.

Instead of directly generating code.

Generate

tables:

- LOAN_MASTER

- CUSTOMER

joins:

- LOAN_MASTER.customer_id =
  CUSTOMER.customer_id

filters:

- PRODUCT_CODE='HL'

- STATUS='S'

- SANCTION_DATE

BETWEEN

2026-05-01

2026-05-31

select:

- CUSTOMER_NAME

- SANCTION_AMOUNT

aggregation:

None

Everything after this becomes deterministic.

PySpark Generator

Prompt

Generate PySpark.

Only use:

These Tables

These Columns

These Joins

These Filters

Don't invent anything.
Validator

Checks

✅ Table exists

✅ Column exists

✅ Join exists

✅ Datatype

✅ Filter value valid

✅ Aggregation valid

If fail

↓

Retry

Runtime Flow
User

↓

"Average Home Loan Amount sanctioned in May 2026"

↓

Intent

↓

Vector Search

↓

Top Columns

↓

DuckDB Metadata

↓

Business Rules

↓

Join Graph

↓

Logical Query Plan

↓

PySpark

↓

Validator

↓

Return Script
Technologies
Layer	Technology
Metadata Store	DuckDB (POC) / PostgreSQL (Production)
Semantic Search	OpenSearch, Qdrant, or Pinecone
Join Engine	NetworkX
Orchestration	LangGraph
LLM	GPT-5.5 / Claude / Bedrock
Execution	PySpark on EMR
Metadata Ingestion	Pandas + SQLAlchemy / DuckDB
Folder Structure
metadata-agent/

├── metadata/
│   ├── metadata.xlsx
│   ├── ingest.py
│   ├── enrich_metadata.py
│   ├── build_graph.py
│   ├── build_embeddings.py
│   └── metadata.duckdb
│
├── vector_db/
│   ├── column_collection
│   ├── table_collection
│   └── mart_collection
│
├── graph/
│   ├── networkx_graph.pkl
│   └── join_service.py
│
├── tools/
│   ├── search_columns.py
│   ├── search_tables.py
│   ├── get_relationships.py
│   ├── resolve_values.py
│   ├── validate_metadata.py
│   └── generate_pyspark.py
│
├── langgraph/
│   ├── state.py
│   ├── workflow.py
│   └── nodes/
│
└── execution/
    ├── spark_executor.py
    └── optimizer.py