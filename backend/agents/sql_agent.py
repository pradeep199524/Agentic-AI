import os
import re
import json
from typing import List, Dict, Any, Union
from sqlalchemy import text
from openai import AsyncOpenAI, AsyncAzureOpenAI

from backend.database import engine 

class PostgresSQLAgent:
    """
    Generic, Read-Only PostgreSQL Agent.
    Fetches full records without restrictive column/row limits so the final LLM can evaluate all data.
    """

    def __init__(self):
        self.schema_name = os.getenv("POSTGRES_SCHEMA", "public")
        self.engine = engine

    def get_live_schema(self) -> Dict[str, Any]:
        """Dynamically inspects and maps out uploaded CSV tables and columns in the active schema."""
        schema = {}
        if not self.engine:
            return schema

        try:
            with self.engine.connect() as conn:
                # Query target data tables only (avoid system tables like pages/documents)
                tables_res = conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema = :schema AND table_name LIKE 'csv_data_%';"),
                    {"schema": self.schema_name}
                )
                tables = [row[0] for row in tables_res]

                for table in tables:
                    cols_res = conn.execute(
                        text("SELECT column_name FROM information_schema.columns WHERE table_name = :table;"),
                        {"table": table}
                    )
                    columns = [row[0] for row in cols_res]

                    # Fetch a sample to show column data types and formats
                    sample_res = conn.execute(text(f'SELECT * FROM "{table}" LIMIT 2;'))
                    sample_rows = [dict(zip(columns, row)) for row in sample_res.fetchall()]

                    schema[table] = {
                        "columns": columns,
                        "sample_records": sample_rows
                    }
        except Exception as e:
            print(f"[WARN Postgres Agent] Schema inspection failed: {e}")

        return schema

    async def execute_query(self, query: str, llm_client: Union[AsyncOpenAI, AsyncAzureOpenAI], model: str) -> List[str]:
        """Translates user query into SQL returning all columns and full records for LLM decision-making."""
        results = []
        if not self.engine:
            return results

        schema_manifest = self.get_live_schema()
        if not schema_manifest:
            return results

        schema_blocks = [
            f"TABLE: {t}\nCOLUMNS: {', '.join(m['columns'])}\nSAMPLE: {json.dumps(m['sample_records'], default=str)}"
            for t, m in schema_manifest.items()
        ]
        dynamic_schema_context = "\n\n".join(schema_blocks)

        system_prompt = (
            "You are an autonomous PostgreSQL database analyst.\n"
            "Given the user question and the dynamic database schema provided below, write a single executable PostgreSQL query.\n\n"
            "STRICT RULES:\n"
            "1. ADAPTIVE QUERIES: If the question explicitly asks for a specific calculation (e.g., 'how many', 'average') or a strict ranking (e.g., 'top 5', 'highest'), use aggregations (COUNT, AVG) and LIMIT appropriately to answer the specific question.\n"
            "2. DEFAULT TO FULL CONTEXT: If the question is general and does NOT ask for a specific count or top/bottom limit, default to `SELECT *` and DO NOT use `LIMIT`. This ensures the evaluation LLM receives the complete tabular context.\n"
            "3. ORDERING: Include `ORDER BY column_name DESC` or `ASC` to organize the data logically, especially for ranking questions.\n"
            "4. NO MARKDOWN: Output ONLY the raw SQL query. Do not wrap in ```sql backticks or add comments.\n"
            "5. SCHEMA INTEGRITY: Use ONLY table and column names that exist in the provided schema.\n"
            "6. IRRELEVANT QUERIES: If the question does not match any table in the schema, output EXACTLY: NONE"
        )

        user_prompt = f"--- DATABASE SCHEMA ---\n{dynamic_schema_context}\n\nUser Question: {query}\nSQL Query:"

        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=250
            )

            raw_sql = response.choices[0].message.content.strip()

            # Skip if the agent determines the question is non-tabular
            if raw_sql.upper() == "NONE":
                return results

            # Strip any accidental markdown formatting
            clean_sql = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE).strip("` \n\r")
            match = re.search(r"(SELECT[\s\S]+)", clean_sql, re.IGNORECASE)
            if not match: 
                return results
            sql_to_run = match.group(1).split(";")[0].strip() + ";"

            print(f"[DEBUG Postgres Agent Executing]: {sql_to_run}")

            # Execute query and format every row with its full set of key-value pairs
            with self.engine.connect() as conn:
                cursor = conn.execute(text(sql_to_run))
                rows = cursor.fetchall()
                
                if rows:
                    keys = list(cursor.keys())
                    formatted_rows = [
                        " | ".join([f"{k}: {v}" for k, v in dict(zip(keys, row)).items() if v is not None]) 
                        for row in rows
                    ]
                    results.append("[PostgreSQL Database Records]:\n" + "\n".join(formatted_rows))

        except Exception as e:
            print(f"[WARN Postgres Execution Error]: {e}")

        return results