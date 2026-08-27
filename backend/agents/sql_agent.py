import os
import re
import json
from typing import List, Dict, Any
from sqlalchemy import text
from openai import AsyncOpenAI

# Import the existing engine directly from your database.py to avoid duplicate connections
from backend.database import engine 

class PostgresSQLAgent:
    """
    Strictly Read-Only PostgreSQL Agent.
    Uses the existing database engine from database.py.
    """

    def __init__(self):
        # Fetch the schema dynamically from environment variables, defaulting to 'public'
        self.schema_name = os.getenv("POSTGRES_SCHEMA", "public")
        
        # Assign the existing engine imported from database.py
        self.engine = engine
        
        if self.engine:
            print("[INFO] SQL Agent successfully connected using the existing database engine.")
        else:
            print("[WARN] Failed to load engine from database.py.")

    def get_live_schema(self) -> Dict[str, Any]:
        """Dynamically maps out tables and columns from the specified PostgreSQL schema."""
        schema = {}
        if not self.engine:
            return schema

        try:
            with self.engine.connect() as conn:
                # Use parameterized query to fetch table names dynamically based on the schema
                tables_res = conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema = :schema;"),
                    {"schema": self.schema_name}
                )
                tables = [row[0] for row in tables_res]

                for table in tables:
                    # Fetch column names for each discovered table
                    cols_res = conn.execute(
                        text("SELECT column_name FROM information_schema.columns WHERE table_name = :table;"),
                        {"table": table}
                    )
                    columns = [row[0] for row in cols_res]

                    # Fetch 2 sample records to provide data formatting context to the LLM
                    sample_res = conn.execute(text(f"SELECT * FROM {table} LIMIT 2;"))
                    sample_rows = [dict(zip(columns, row)) for row in sample_res.fetchall()]

                    schema[table] = {
                        "columns": columns,
                        "sample_records": sample_rows
                    }
        except Exception as e:
            print(f"[WARN Postgres Agent] Schema inspection failed: {e}")

        return schema

    async def execute_query(self, query: str, llm_client: AsyncOpenAI, model: str) -> List[str]:
        """Translates the user query into SQL, executes it, and returns the formatted rows."""
        results = []
        if not self.engine:
            return results

        schema_manifest = self.get_live_schema()
        if not schema_manifest:
            return results

        # Construct the schema context for the LLM prompt
        schema_blocks = [
            f"TABLE: {t}\nCOLUMNS: {', '.join(m['columns'])}\nSAMPLE: {json.dumps(m['sample_records'], default=str)}"
            for t, m in schema_manifest.items()
        ]
        dynamic_schema_context = "\n\n".join(schema_blocks)

        system_prompt = (
            "You are an autonomous PostgreSQL database analyst.\n"
            "Given the user question and the dynamic database schema provided below, write a single executable PostgreSQL query.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the raw SQL statement. Do NOT include markdown fences (```), backticks, explanations, or comments.\n"
            "2. Use ONLY the table names and column names present in the provided schema context. Do not invent or guess names.\n"
            "3. Always start your query with `SELECT *` from the matching table to preserve full row context for the answer synthesizer.\n"
            "4. For string/text matching on columns, use case-insensitive matching with PostgreSQL's ILIKE operator (e.g., WHERE column_name ILIKE '%keyword%').\n"
            "5. Cast numerical or other non-text columns to TEXT if performing keyword pattern matching: WHERE column_name::text ILIKE '%keyword%'.\n"
            "6. For ranking or extreme queries (e.g., highest, lowest, top), use ORDER BY <column_name> DESC/ASC LIMIT 1.\n"
            "7. If no table or column in the schema can answer the user question, output EXACTLY: NONE\n\n"
            "SYNTAX FORMAT:\n"
            "SELECT * FROM <matched_table> WHERE <matched_column> ILIKE '%<search_term>%';"
        )

        user_prompt = f"--- DATABASE SCHEMA ---\n{dynamic_schema_context}\n\nUser Question: {query}\nSQL Query:"

        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=250
            )

            raw_sql = response.choices[0].message.content.strip()

            # Fallback check: if LLM determines the schema cannot answer the query
            if "NONE" in raw_sql.upper() or not raw_sql.upper().startswith("SELECT"):
                # Regex fallback to extract SQL if the LLM accidentally includes markdown
                clean_sql = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE).strip("` \n\r")
                match = re.search(r"(SELECT[\s\S]+)", clean_sql, re.IGNORECASE)
                if not match: 
                    return results
                raw_sql = match.group(1)

            # Ensure the query ends with a semicolon and execute
            sql_to_run = raw_sql.split(";")[0].strip() + ";"
            print(f"[DEBUG Postgres Agent Executing]: {sql_to_run}")

            with self.engine.connect() as conn:
                cursor = conn.execute(text(sql_to_run))
                rows = cursor.fetchall()
                
                if rows:
                    keys = list(cursor.keys())
                    # Format the database rows into readable strings for the final LLM context
                    formatted_rows = [", ".join([f"{k}: {v}" for k, v in dict(zip(keys, row)).items()]) for row in rows[:25]]
                    results.append("[PostgreSQL Database Records]:\n" + "\n".join(formatted_rows))

        except Exception as e:
            print(f"[WARN Postgres Execution Error]: {e}")

        return results