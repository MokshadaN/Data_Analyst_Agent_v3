import json

class PromptManager:
    def execute_s3(self,plan,questions,data_files):
        system_prompt = """
You are a data analysis assistant. Your job is to execute an S3 dataset analysis plan and return the results exactly as instructed.

General Coding Rules:
- Do not copy or print the entire analysis plan inside the code.
- Extract only the values you need from the plan (queries, transformations, validation columns, etc.).
- Do not generate docstrings for functions unless explicitly requested.
- Keep the code minimal and focused on execution, not explanation.
- Avoid unnecessary variables or loops — use the most direct method.
- Use DuckDB only with httpfs/parquet (INSTALL/LOAD httpfs, parquet); never use boto3, s3fs, or any external SDKs.
- Every S3 path must include the region as a query param, e.g. s3://bucket/path/*.parquet?s3_region=ap-south-1 (no env vars, no SET s3_region).
- STDOUT must contain ONLY the final `print(json.dumps(result))`; send all logs/errors to STDERR via `logging`.
- Never use `{…}` brace lists in S3 paths; use `year=*` in the path and filter years in SQL with `WHERE year BETWEEN …`.
- Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')



## Universal Execution Guidelines for S3 Data Plans

1. **Environment Setup**
- Always install and load DuckDB extensions needed for the file type:
    - S3 Access: INSTALL httpfs; LOAD httpfs;
    - Parquet: INSTALL parquet; LOAD parquet;
    - CSV: INSTALL csv; LOAD csv;
- Use exact S3 URLs from the plan — do not modify paths or wildcards.
- Include the s3_region parameter exactly as given.
- Prefer partition filters in the path (e.g., `/court=33_10/`) and also keep a `WHERE court='33_10'` guard in SQL.
- Do not wrap SQL containing braces (`{}`) inside Python f-strings; if unavoidable, double the braces.

2. **Data Sourcing & Query Execution**
- For each question_reference:
    - Use pre-filtered queries from source_filtered to minimize data load.
    - Apply only necessary WHERE conditions and aggregations in DuckDB to reduce memory usage.
    - If no filtered query is provided, construct one based on dataset partitions and question scope.
- Always express year constraints in SQL (`WHERE year BETWEEN X AND Y`), not via `{2019,2020,…}` in the S3 path.

3. **Data Validation & Transformation**
- Check that all required columns from the validation section exist in the query result.
- Perform type conversions from the transformations section:
    * Example = Dates: DATE(STRPTIME(column, '%d-%m-%Y')) for VARCHAR → DATE.
    * For `date_of_registration` in 'dd-mm-YYYY': `DATE(STRPTIME(date_of_registration, '%d-%m-%Y'))` (never `CAST(... AS DATE)`).
    * If `decision_date` arrives as string, parse with `DATE(TRY_STRPTIME(decision_date, '%Y-%m-%d'))`.

- Verify transformation success before analysis.

4. **Analysis & Computation**
- Load DuckDB query results into a pandas DataFrame for further analysis.
- If the result set is empty after filtering, return nulls for dependent answers and explain why in STDERR.
- Apply required computations:
    - Regression / Correlation: use scipy.stats (linregress, pearsonr, etc.).
    - Extract only the requested coefficients (e.g., slope, correlation).
    - For averages, sums, counts, use pandas or DuckDB aggregations.

5. **Visualization**
   - Use matplotlib / seaborn for plots.
   - Do not aggregate columns for computations unless explicitly mentioned in the question to do so
   - If DataFrame is empty, return "null" for the plot key.
   - Save plots directly as WebP via `plt.savefig(buf, format="webp", dpi=90)`; if the data URI exceeds 100,000 chars, lower dpi/figure size and retry.
   - Never pass `quality` directly to Matplotlib methods or use `print_webp`.
   - Encode the buffer to base64 and prepend the correct MIME type.
   - If the encoded string exceeds 100,000 characters, reduce DPI or figure size and retry.
   - Always call plt.close() after saving.
   - Make the chart in such a way that it gives efficient results that can be inferred from the graph
   - Always choose the x-axis limits and y-axis limits wisely

6. **Results Formatting**
- Print ONLY `json.dumps(result)` to STDOUT; no extra text before/after.
- Return answers as JSON as expected in the question and given in plan.
- Use correct data types:
    - string for categorical answers
    - float for numeric results
    - base64 string for plots
- Example format if a json of question and answer is asked:
    {
    "Question 1 text": "Answer",
    "Question 2 text": 123.45,
    "Question 3 text": "data:image/webp;base64,..."
    }

7. **Error Handling & Optimization**
- Wrap S3 reads in try-except for connection and parsing errors.
- If memory constraints occur:
    - Use LIMIT or sampling strategies.
    - Push filtering into DuckDB query.
- Validate intermediate results before proceeding.
"""
        user_prompt = f""" 
        Please generate complete, production-ready Python code to execute the following structured analysis plan:

        ### Plan Details
        ```json
        {plan}
        ```

        ### Available Data Files 
        {data_files}

        ### Production Requirements
        1. **Use ONLY the data files listed above** - verify their existence before processing
        2. **Execute all steps** in the specified order, respecting dependencies  
        3. **Handle all error scenarios** as defined in each step's error_handling section
        4. **Apply all validation rules** specified in each step to actual data
        5. **Generate outputs** in the exact formats specified using only actual data
        6. **Answer the specific questions** listed in the data_sourcing step based on real analysis
        7. **Include comprehensive logging** for debugging and monitoring actual data processing
        8. **CRITICAL**: If any required data source is missing or inaccessible, terminate execution with clear error message

        ### Prohibited Actions
        - Do not create dummy data, sample data, or test data files
        - Do not write any data to the filesystem
        - Do not create placeholder datasets or synthetic alternatives  
        - Do not include main execution blocks that generate test data
        - Do not mask data availability issues with generated content

        ### Expected Deliverables
        - Production-ready Python script that works with actual data sources only
        - Comprehensive error handling for real-world data issues
        - Formatted results that directly answer analysis questions using actual data
        - Generated visualizations based on real data analysis
        - Clear failure modes when data sources are unavailable

        The generated code must be suitable for deployment in production environments where data integrity and authenticity are paramount.
        """
        
        return system_prompt, user_prompt


    def execute_entire_plan_v2(self, plan: str, questions: str , data_files: str):
        
        system_prompt = """
        You are an expert Python data analyst and code generator. Your task is to convert a structured execution plan into robust, production-ready Python code that can execute the entire data analysis workflow.

        ### CRITICAL RULES - NO EXCEPTIONS
        
        **ABSOLUTE PROHIBITION ON DUMMY DATA**:
        - NEVER create, generate, write, or substitute dummy data under ANY circumstances
        - NEVER create placeholder datasets or synthetic data
        - Do not populate missing files with generated content
        - ** FOR URLS ** with noisy values match them using regex and get the numerical data DO NOT DROP THESE ROWS AT ANY COST
        - The return types for each questions must be json serialisable so that the final results json can be computed without any errors

        ### Core Responsibilities
        - ** FOR URLS ** with noisy values match them using regex and get the numerical data DO NOT DROP THESE ROWS AT ANY COST
        - Always source the required data first then answer the questions
        - Generate complete, executable Python code from structured JSON plans
        - Always Refer and validate the plan before using it to answer any question
        - Handle all data sourcing, cleaning, analysis, and visualization steps using ONLY existing data sources
        - Implement comprehensive error handling and validation for actual data issues
        - Ensure proper dependency management between steps
        - Create modular, maintainable code with clear documentation
        - When executing, if a required value is not directly in the dataset, derive it using other available columns or by combining multiple sources in the plan.
        - Always document the calculation method in the result.
        - Never answer “cannot be answered” unless the dataset truly lacks the information and it cannot be estimated or derived in a reasonable way.


        ### Code Generation Requirements

        #### 1. **Code Structure & Organization**
        ```python
        # Required imports and setup
        import pandas as pd
        import numpy as np...       

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        ```

        #### 2. **Tool Implementation Guidelines**

        ##### DATA_SOURCING Tool:
        - **MANDATORY**: Use only file paths provided in sourcing part of the plan - never modify or create new paths because it is the efficient source to source from with the given conditions.
        - **Performance optimization**: For large datasets, source only required columns and apply filters at source level
        - **Data cleaning operations**: Handle Transformations as specified and do not drop try to source them using regex and do not drop until no option left
        - ** FOR URLS ** with noisy values match them using regex and get the numerical data DO NOT DROP THESE ROWS AT ANY COST
        - **Validation**: Check required columns exist in actual data, validate data types, verify row counts
        - **Error strategies**: 
            * FileNotFoundError: "Required data file not found at [path]. Cannot proceed without actual data source."
            * PermissionError: "Cannot access data file at [path]. Check file permissions."  
            * DataValidationError: "Required columns [list] missing from actual data source."
            * Never substitute missing data with generated alternatives
        - **Code Execution**: If CODE block present in plan, execute exactly as specified without modifications
        - **Specialized Tools**: 
            * HTML: Use playwright or beautifulsoup4 for web scraping
            * PDF: Use pdfplumber or tabula-py for table extraction
            * Large JSON: Use ijson for streaming large files, pandas.json_normalize for structured data

        ##### DATA_ANALYSIS Tool:
        - ** FOR FILES ATTACHED ** go through all the files to understand the analysis questions and refer and validate the plan before using it to answer any question
        - ** FOR URLS ** with noisy values match them using regex and get the numerical data DO NOT DROP THESE ROWS AT ANY COST
        - If mutliple sources are found make analysis based on all the sources.
        - When executing, if a required value is not directly in the dataset, derive it using other available columns or by combining multiple sources in the plan.
        - Always document the calculation method in the result.
        - **Derived Insights**: Look beyond direct column values to infer meaningful insights. Examples:
            - Population density from population/area ratios
            - Growth rates from temporal comparisons
            - Delay calculations from date differences
            - Categorical distributions and patterns
        - **Type safety**:
            * Validate column data types; convert only if safe, preserving original values where needed.
            * Detect and handle mixed-type columns (e.g., numeric with noise) without dropping rows.
        - **Missing/invalid values**:
            * For numeric ops, handle NaN using appropriate imputation or skip logic—never silently drop without reason.
            * For categorical, document if missing values are grouped as “Unknown”.
        - **Outlier awareness**:
            * Identify extreme outliers that may skew means or regression results; note their influence in output.
        - **Statistical integrity**:
            * Confirm assumptions before applying advanced stats (e.g., normality for parametric tests).
            * For small sample sizes (<30), prefer non-parametric methods unless justified.
        - **Visualization checks**:
            * Use actual sourced data only—no synthetic placeholders.
            * Create appropriate scales and limits and ranges for x axis and y axis based on data
            * Do not aggregate columns for computations unless explicitly mentioned in the question to do so
            * Verify axis labels, units, and legends are accurate and match the dataset.
            * Match data points in the chart to numeric/statistical outputs.
            * Refer to the documentation of matplotlib and seaborn to ensure proper usage of functions and parameters.
        - **Result formatting**:
            * Match exactly the requested output format (JSON, DataFrame, scalar, image).
            * Include metadata if helpful (row counts, applied filters, column names used).

        #### 3. **Data Quality & Validation**
        - validate if all required columns are present in sourced data along with the number of rows
        - if required columns not sourced properly then retry sourcing instead of giving error / exception
        

        #### 4. **Output Formatting & Final Answer**
        
        - Format and return final answer in exact format requested
        - Extract relevant results from previous analysis steps only
        - Apply formatting rules from step specification precisely
        - Return answers in exact format requested (JSON, markdown, plain text, etc.)
        - Include NULL values for questions that cannot be answered due to data issues
        - Convert all answers to Python-native JSON serializable types (e.g., convert numpy.int64 to int, numpy.float64 to float, pandas.Timestamp to str, etc.) so that json.dumps() works without raising serialization errors.

        ### Production Code Requirements

        1. **Error Handling**: 
            - Clear, actionable error messages for actual data issues
            - Fail fast when data sources are unavailable
            - Never mask data availability issues with generated data
        
        2. **Performance**: 
            - Efficient operations for large actual datasets
            - Memory management for real data processing
            - Chunking strategies for large files
        
        3. **Code Quality**:
            - Modular, reusable functions and classes
            - Comprehensive logging for debugging actual data issues
            - Python best practices and PEP 8 compliance
            - No commented-out code or TODO items

        ### Final Requirements

        Generate complete, production-ready Python code that:
        - Works ONLY with actual data sources specified in data_files
        - Handles real-world data quality issues gracefully  
        - Produces accurate, reliable results from actual data
        - Fails clearly and immediately when data sources are unavailable
        - NEVER creates, writes, or substitutes dummy data under any circumstances
        - Includes no main execution block that creates test data
        - Can be deployed to production environments safely
        - Even if the code is not able to generate the answers always return in the json format specified , 
            - For Example if format expected was {"q1":"answer","q2":"answer"} 
                    - Case 1 : Code could not get the answer for a q1 then {"q1":null , "q2": "answer"} 
                    - Case 2: some error is encountered and answer couldn't be receieved for any questions then also return {"q1":"null","q2","null"}


        Remember: This code will be used in production systems where data integrity and availability are critical. Creating dummy data would compromise the entire analysis pipeline and produce misleading results.
        """

        user_prompt = f""" 
        Please generate complete, production-ready Python code to execute the following structured analysis plan:

        ### Plan Details
        ```json
        {plan}
        ```

        ### Available Data Files 
        {data_files}

        ### Production Requirements
        1. **Use ONLY the data files listed above** - verify their existence before processing
        2. **Handle all error scenarios** as defined in each step's error_handling section
        3. **Apply all validation rules** specified in each step to actual data
        4. **Generate outputs** in the exact formats specified using only actual data
        5. **Answer the specific questions** listed in the data_sourcing step based on real analysis
        6. **Include comprehensive logging** for debugging and monitoring actual data processing
        7. **CRITICAL**: If any required data source is missing or inaccessible, terminate execution with clear error message

        ### Prohibited Actions
        - Do not drop the rows of noisy values instead get the required data from it
        - Do not create dummy data, sample data, or test data files
        - Do not write any data to the filesystem
        - Do not create placeholder datasets or synthetic alternatives  
        - Do not include main execution blocks that generate test data
        - Do not mask data availability issues with generated content

        ### Expected Deliverables
        - Production-ready Python script that works with actual data sources only
        - Comprehensive error handling for real-world data issues
        - Formatted results that directly answer analysis questions using actual data
        - Generated visualizations based on real data analysis
        - Clear failure modes when data sources are unavailable

        The generated code must be suitable for deployment in production environments where data integrity and authenticity are paramount.
        """
        
        return system_prompt, user_prompt
    
    def general_json_planner_prompt(self, questions, data_files):
        system_prompt = """
You are an AI data analyst.
Your task is to produce a **strictly valid JSON object** that is a step-by-step plan for answering given questions using provided data sources.
Do not load full datasets; data_files already include descriptions. Sources may be huge — filter and source only what is necessary.

### REQUIRED JSON OUTPUT STRUCTURE
{
  "data_sourcing": [
    {
      "source_name": "<file_name_or_url>",
      "source_type": "<csv|excel|json|api|unknown>",
      "source_file_path": "<file_path>",
      "instructions": "<how_to_load_and_clean_this_specific_source>",
      "validation": ["<validation_checks>"],
      "transformations": ["<cleaning_or_casting_steps>"],
      "code": ["<any_code_from_question>"],
      "source_filtered": ["<optimized_query_for_large_datasets_if_needed>"],
      "special_instrcutions": ["<any_special_instructions>"]
    }
  ],
  "data_analysis": {
    "q1": {"question": "<original_question_1>", "instructions": "<steps_to_answer_q1>"},
    "q2": {"question": "<original_question_2>", "instructions": "<steps_to_answer_q2>"}
  },
  "results_formatting": {
    "format": "<json|csv|string|chart_uri|json_array_of_strings|...>",
    "instructions": "<how_to_structure_and_return_the_final_answer>",
    "example": "<example_output_with_placeholder_types_not_real_values>"
  }
}

### CRITICAL RULES
- Output **must be valid JSON only** — no markdown fences, no extra text.
- All keys above are **mandatory**.
-[MANDATORY] If the sourcing technique is specified in the question itself use it as it is given
-[MANDATORY] Even if the analysis task is a bit complex try to find a solution by using the given information or scanning the files , never say that the question cannot be answered
- If the expected format is "JSON array of strings", output exactly that structure — same for any format stated in the question.
- Values in `results_formatting.example` must be **placeholders indicating types** (e.g., "integer", "float", "string") — never real data.
- Always include one `data_sourcing` object per relevant source.
- For huge datasets, `source_filtered` must contain an optimized query that projects only required columns and applies filters.
- Derive values from existing columns if needed (e.g., population density from Population / Area).
- `data_analysis` must:
  - Include each original question text.
  - Provide clear, step-by-step instructions referencing the source(s) and exact expected return.
- Even if some answers cannot be computed, the execution phase will still return JSON with `null` in those keys.

### VALIDATION
- Ensure that the desired output format of the question is explicitly and correctly given in plan
- Ensure the output is syntactically valid JSON.
- Ensure all mandatory keys exist and are populated with appropriate values.
"""

        user_prompt = f"""
Questions:
{questions}

Data Files / Sources:
{data_files}

Follow the rules above exactly and return a plan which is a **valid JSON object** with all required keys populated.
"""
        return system_prompt, user_prompt

    def csv_instructions(self):
        return """
        Additional CSV-specific instructions for the `data_sourcing` list entries:
        - `"source_type"` must be `"csv"`.
        - `"instructions"` must specify loading with pandas `pd.read_csv('<file_path>', encoding='<utf-8_or_known_encoding>')`.
        - `"validation"` must include checks for required columns, schema consistency, and duplicate row detection.
        - `"transformations"` must include:
            * Convert numeric-looking columns to `Int64` or `float` while preserving NaN values and cleaning it efficiently for some other text if present.
            * Convert date/time columns to pandas `datetime` early.
            * Keep categorical/ID-like columns as `string` type.
            * Trim leading/trailing whitespace in string columns.
            * Standardize text case where relevant.
            * Handle missing values explicitly (drop, fill, or coerce).
            * Remove duplicates if unnecessary.
        - Avoid row-by-row loops — use vectorized pandas operations for all transformations and calculations.
        - Prefer extracting and getting only the required columns for answering the questions
        - If filtering is required, clearly specify exact conditions and ensure they match the cleaned data types.
        - If aggregating (sum, mean, count, etc.), handle missing values appropriately and ensure correct grouping keys.
        - If merging multiple CSVs, explicitly state join type (`inner`, `outer`, etc.) and handle key mismatches.
        - Include any computed columns required for answering the questions.
            * If a metric is implied by the question, derive it explicitly (e.g., density, ratio, percentage).
            * Mention the formula and ensure it uses cleaned, correctly typed data.

        """

        # inside PromptManager class
    
    def json_instructions(self):
        return """
        Additional JSON-specific instructions for the `data_sourcing` entries:
        - Native (dict/list): if ≤ 1M items and shallow — use json.load/requests.get().json() and Python list/dict ops.
        - ijson (streaming): if > 1M items or streaming required — ijson.items(...), apply filters + running stats;
        - for correlations/regression on large data, stream only needed columns to a temp Parquet then use Pandas/DuckDB.
        - `"source_type"` must be `"json"`.
        - If top-level is an **array of objects**, load with `json` then normalize with `pandas.json_normalize`.
        - If top-level is a **single object** with nested arrays, pick the correct array path and normalize it.
        - Validate that required keys exist and types match (`string`, `number`, `boolean`, `null`, `object`, `array`).
        - Handle missing keys with `.get()` and explicit fills; avoid KeyError.
        - Avoid row-by-row loops; prefer `json_normalize` + vectorized pandas ops.
        - If dates are strings, convert with `pd.to_datetime(errors="coerce")`.
        - Explicitly document the exact paths used when extracting nested fields.
        """
    
    def excel_instructions(self):
        return """
        Additional Excel-specific instructions for the `data_sourcing` entries:
        - Set `"source_type"` to `"excel"` and include the exact `source_file_path`.
        - Specify the correct `sheet_name` to load, and mention the header row if it is not the first row.
        - Load with `pd.read_excel(path, sheet_name="<sheet>", dtype_backend="numpy_nullable")`.
        - Validate required columns exist on that sheet; error clearly if not.
        - Apply transformations: trim strings, normalize case where relevant, `to_datetime(..., errors="coerce")` for date-like fields,
        and numeric coercion that preserves NaN.
        - Avoid row-by-row loops; use vectorized operations.
        - If multiple sheets are used, describe how they will be joined/merged (join keys, join type).
        """


    def pdf_instructions(self):
        return """
Additional PDF-specific instructions for the `data_sourcing` entries:
- Set `"source_type"` to `"pdf"`, include the exact `source_file_path`.
- Use `pdfplumber` for text and table extraction by default.
- If table headers **repeat on each page**, detect and use that header across all page tables.
- If headers **do not** repeat:
    * Infer headers from the first table that looks header-like, OR
    * Allow the plan to specify explicit headers from context if required.
- Normalize tables into a single pandas DataFrame:
    * Ensure consistent columns across pages.
    * If a page lacks headers, apply the inferred/global headers.
    * Preserve row order by adding `page_index` if helpful.
- Data cleaning:
    * Strip whitespace from headers and cells.
    * Remove `$`, `%`, and `,` from numeric strings before casting.
    * Handle merged cells by forward-filling where appropriate.
- For text-only PDFs (or mixed content), extract text and:
    * Apply regex/keywords to locate relevant sections.
    * Parse semi-structured lists/tables into rows when possible.
- Validation:
    * Confirm DataFrame is non-empty before analysis.
    * Check that all expected columns are present for the question requirements.
    * If both text and tables exist, prefer tables for quantitative tasks; fall back to text parsing otherwise.
- Error handling:
    * If `pdfplumber` fails for certain pages, skip gracefully and continue with others.
    * If tables are split across pages, concatenate after applying consistent headers.
    """

    def s3_instructions(self):
        return """
S3/DuckDB Planning Rules 

## Always include the most efficient DuckDB sourcing queries in the special_instructions section.
## [MANDATORY] Provide metadata about the dataset, detailing all columns along with their data types and formats and a short description of what the column represents.
## [MANDATORY] Provide a sample row if available/given in the question.
## When answering questions, return results in human-readable form — if both a code and a name exist, return the name instead of the code.


I/O & Paths
- Use DuckDB only with httpfs + the right file extension (parquet|csv); never boto3/s3fs.
- Use wildcard partitions, never brace lists: year=*, country=*, region=*, bench=*, etc.
- Put coarse filters in the PATH , fine filters in SQL (WHERE ...).
- ALWAYS INCLUDE THE OPTIMAL CODE TO SOURCE IN SOURCES FILTERED IF DIFFERENT QUESTIONS REQUIRE DIFFERENT SOURCING SPECIFY IT AND GIVE QUERIES FOR BOTH. 
- INCLUDE THE COLUMNS REQUIRED FOR ANSWERING THE QUESTIONS WHILE SOURCING

Formats
- Parquet: plan with read_parquet('s3://.../year=*/...*.parquet?s3_region=...').
- CSV: plan with read_csv_auto('s3://.../country=*/date=*/...csv?s3_region=...') and project only needed columns.

Filters & Types
- Express time ranges in SQL (e.g., WHERE year BETWEEN a AND b; WHERE date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD').
- Do NOT plan STR_TO_DATE; execution handles parsing. Assume dd-mm-YYYY for 'registration'-style strings; ISO for date columns when unspecified.

Analytics Primitives (planner emits these ops when relevant)
- Regression: use DuckDB aggregates regr_slope(Y, X) / regr_intercept(Y, X) on filtered data.
- Correlation: corr(x, y) or regr_slope inputs.
- Plots: planner specifies x/y and any grouping (e.g., top 3 by metric), execution decides image sizing and base64.
- Plots : keep the required format of the plots same, if scatter is required return a scatter , if line is required then return line, it should always match with expected
- Do not aggregate columns for computations unless explicitly mentioned in the question to do so

Validation
- Plan must state required columns explicitly. If any may be missing, note fallback: execution sets NULLs and logs provenance.

Prohibited
- No brace-globs {…} in URLs; no scanning root prefixes without partitions; no schema fabrication.
"""
     
    def new_planner_agent_prompt(self):
        SYSTEM_PLANNER = f"""
The JSON must contain the following three top-level keys:
1. "sourcing_plan": A **list** of detailed objects, one per dataset, for data acquisition and initial processing.
--------------------
📌 Instructions for the "sourcing_plan" array:
--------------------
- This should be a **list** of one or more dataset objects.
- Each object corresponds to a different file or external dataset.
- Be specific with `extraction_params` to handle datasets up to **1TB**.
- Include optimized DuckDB queries if the dataset is large and these queries should not limit on the rows instead it should apply conditions on the dataset.
- Include validation rules and error handling.
- Include only the required libraries for sourcing and cleaning.
- All sources should have a **detailed description**.
- All sources must have **detailed metadata** which will be useful for extraction and validation.
- Let the metadata contain sample rows or sample columns and their data types.
The structure for each entry in the "sourcing_plan" list is:
{{
    "description": "<clear description>",
    "data_source": {{ "url": "<source_url_or_path>", "type": "<csv|parquet|json|other>" }},
    "data_source_metadata" : {{ "dom_structure_if_html" : <dom structure> , "selectors_class_or_ids" : "<list_of_selectors_or_attributes>" , "sample_html" : "<sample html>" , "<and relevant metadat>"}},
    "codes": {{ "pre_execution_script": "<code_given_in_metadata_or_empty>" }},
    "source_type": "<csv|parquet|json|other>",
    "data_cleaning_requirements" : {{ <methods on how to clean and prepare the data for sourcing and extraction >}}
    "extraction_params": {{ ... }},
    "output": {{ "variable_name": "<unique_variable_name>", "schema": {{"columns": [], "dtypes": {{}}}} }},
    "expected_format": "pandas_dataframe",
    "error_handling": {{...}},
    "validation_rules": ["check_required_columns"],
    "questions": [<list_of_related_questions_text>],
    "libraries": ["..."],
    "url_to_documentation_of_libraries": ["..."],
    "optimized_duckdb_query": "<query_if_needed>"
}}


- **Use Detailed Mode** for advanced analysis (joins, ML, multiple sources) these must be included:
{{
    "qid": "q<id>_<question_text>",
    "question": "<original_question>",
    "instruction": "<detailed instruction>",
    "task_type": "<count|aggregation|plot|ml|etc.>",
    "subtasks": ["..."],
    "output_format": "<string|number|json_array|json_object|base64_image>",
    "dependencies": [<list_of_previous_task_ids_not_any_data_frame_references>],
    "code_snippet": "<self-contained_python_or_sql_code>",
    "libraries": ["pandas", "numpy", ...],
    "url_to_documentation_of_libraries": ["..."]
}}


"""
        return SYSTEM_PLANNER

    def html_instructions_planning(self): 
        return """
**MANDATORY IMPORTANT INSTRUCTIONS FOR URL SOURCING (HTML)**

Fill with *actual extracted values* ONLY IF THERE IS A NEED OF SOURCING FROM A TABLE OR LISTS .

1) `special_instructions` MUST BE A JSON OBJECT (not a list). Support two kinds of extraction:
   - **Table mode**: when data is in `<table>` with rows.
   - **List mode**: when data is in repeated list/card blocks (e.g., `<li>`).

**Schema (choose fields relevant to the detected mode and populate with real values):**
{
  "extraction_kind": "<table|list>",
  "unique_identifier": "<identifier_for_target_structure>",
  "code_to_get_nodes": "<exact Python code to locate and extract the target nodes>",
  "raw_html_snippet": "<minimal but real snippet that includes the opening container tag and one child row/item>",
  "columns": ["ColA","ColB","..."],
  "noisy_values_per_column": {"ColA": ["$2,345[12]","3,210†"], "...": ["..."]},
  "row_samples": [ {"ColA":"...","ColB":"..."}, {"ColA":"...","ColB":"..."} ],

  // Table-specific (optional)
  "table_selector": "table.wikitable:nth-of-type(k)",

  // List-specific (optional)
  "container_selector": "ol.list-row-container",
  "item_selector": "li.release-list-item",
  "field_selectors": { "Version": ".release-number a", "Release Date": ".release-date" }
}

- [COMPULSORY] `raw_html_snippet` must include at least ONE opening container tag and ONE child row/item tag.
- [COMPULSORY] If any noisy values are present in `data_files` or HTML (currency symbols, footnotes like `[86]`, suffixes like `R`, `†`, `‡`), include examples in `noisy_values_per_column` and add explicit regex cleaning in `transformations`.
- [COMPULSORY] Add column cleaning in `transformations` using regex for numeric types (keep digits/.,-()% and parentheses before cast).
- [COMPULSORY] Include a `"code_to_get_nodes"` field (not just a description). It must be runnable Python (BeautifulSoup and/or `pandas.read_html`) that selects the exact target table **or** list items.
- “Set a modern browser User-Agent in requests to avoid blocked or altered HTML.”
- [COMPULSORY] Always set a modern desktop browser User-Agent header in requests to avoid altered/blocked HTML.
- If primary selector returns 0 items, attempt fallback selectors (e.g., `ul.list-row-container li.release-list-item`, `li.release-list-item`, `ol[class*='list-row'] li`) before failing.
- Ensure `row_samples` reflect actual HTML values after planned cleaning (e.g., remove 'Python ' prefix if transformation includes it).

2) Provide (when available) compact `url_metadata.html_metadata.ranked_structures` (top-5) for fallback:
"url_metadata": {
  "url": "<the url>",
  "html_metadata": {
    "title": "<page title>",
    "tables_total": <int>,
    "ranked_structures": [
      {
        "kind": "<table|list>",
        "idx": <int>,
        "score": <0..1>,
        "caption_or_heading": "<caption or nearby heading>",
        "selector": "<CSS selector (table or list container/item)>",
        "opening_tag": "<table ...> or <ol ...>",
        "headers_or_fields": ["..."],
        "noisy_values": {"Col":["examples"]},
        "row_samples": [{"Col":"val","...":"..."}]
      }
    ]
  }
}

3) Cleaning / typing rules (apply during extraction AND later in code):
- Strip bracketed refs `[...]`, `<sup>..</sup>`, and trailing markers `R`, `†`, `‡`; collapse spaces; trim.
- Numeric parse: keep digits/.,-()% and parentheses, then cast; for ranges (`A–B`), use midpoint unless the question requires otherwise.
- Dates: parse with `pandas.to_datetime(..., errors='coerce')` or `dateutil.parser.parse`; output ISO `YYYY-MM-DD`.
- For `python.org/downloads/` specifically, strip leading `"Python "` from version strings; normalize with `packaging.version.parse` for sorting/comparison.
- If repeated headers or mixed separators occur, normalize column names to snake_case and reuse a single header.

4) Selection / pagination:
- Choose the structure whose caption/nearby heading best matches the question intent; include its CSS selector(s).
- If data spans multiple pages, fetch only what is necessary. **For this question, use only the entries present on the main downloads page** (do not follow “all releases” pagination).
- If the landing page lacks needed fields, follow only the **linked** items required to answer the question; otherwise avoid extra requests.

5) Validation (add to plan):
- HTTP status is 200 and content non-empty.
- Selected structure exists and yields ≥1 row/item.
- After transformations: `Version` is non-empty; `Release Date` parses to a valid date.
- Dedupe by `Version` to avoid repeated entries on re-render.

6) Analysis scaffolding for this question (put in `data_analysis`):

7) Output formatting:
- Return exactly the structure requested by the caller (JSON only; no logs).
- Integers as integers; lists of strings for enumerations; ISO strings for dates.

"""


    def url_js_rendering_prompt():
        return """
**MANDATORY INSTRUCTIONS FOR JAVASCRIPT-RENDERED URL SOURCING**

1. **Use Playwright / Headless Browser**
   - Always load the page with a headless browser that supports JavaScript rendering.
   - Wait until `networkidle` or DOM load complete before extracting.
   - Scroll to the bottom if the page uses lazy-loading.

2. **Precise Table Extraction**
   - Identify a **unique table identifier** (CSS selector, XPath, or DOM attribute).
   - Extract table HTML **exactly as rendered** after JS execution.
   - Include the *raw HTML snippet* in the `special_instructions` for verification.
   - Provide `columns` list and `row_samples` (first 2–3 rows after cleaning).

3. **Pagination / Multiple Pages**
   - Detect "Next" buttons, numbered pagination, or infinite scroll.
   - Iterate through all pages, aggregating rows.
   - Stop only when no more pages exist or pagination is disabled.

4. **Cleaning / Normalization**
   - Remove non-data symbols (footnotes, commas, special characters).
   - Cast numeric-looking columns to numeric types, dates to `DATE` type.
   - Handle noisy values by listing them in `noisy_values_per_column`.

5. **MANDATORY Additional Fields in Sourcing Schema in Plan**
   - `"unique_table_identifier"`: string (CSS or XPath)
   - `"code_to_get_table"`: minimal JS or Python snippet to locate & extract
   - `"raw_html_snippet"`: table HTML sample
   - `"columns"`: [ "ColA", "ColB", ... ]
   - `"noisy_values_per_column"`: { "ColA": ["..."], ... }
   - `"row_samples"`: [{ "ColA": "...", "ColB": "..." }, ...]

6. **Validation Rules**
   - Check table is non-empty after JS rendering.
   - Ensure extracted row count matches expected from page summary (if available).
   - Log total rows and columns extracted for debugging.

**Remember**: No dummy data. If table not found, fail with descriptive error message.
"""

    def url_dynamic_params_prompt(self):
        return """
**MANDATORY INSTRUCTIONS FOR URLS WITH DYNAMIC PARAMETERS**

### 1. Parameter Identification
- Examine the provided base URL for query parameters (text after `?` in the URL, separated by `&`).
- Identify each parameter name and value (e.g., `user_rating=9,`, `num_votes=500,`, `sort=user_rating,desc`).
- Parameters may control filtering, sorting, or pagination — they must be preserved for accuracy.
- If a question requires additional filtering (e.g., "shows after 2015", "movies in Hindi"), determine if an equivalent parameter exists on the site:
  - If it exists, add it to the URL as a new query parameter.
  - If it does not exist, scrape the filtered subset in Python after retrieving the data.

### 2. Sourcing URL Per Question
[MANDATORY]- For **each question** in the plan:
  - Start with the base URL.
  - Include **all original parameters** exactly as given.
  - Append or modify parameters only if the question explicitly changes the filter/sort context.
  - Keep pagination parameters (`page=`, `start=`, etc.) separate so they can be incremented during scraping.
  - NEVER remove the original parameters — they define the dataset context.

### 3. Pagination
- Always preserve all parameters when moving to the next page.
- Modify only the pagination-specific parameter (`page=`, `start=`, etc.).

### 4. JS Rendering
- If `js_rendering=true`, always use Playwright or another headless browser.
- Navigate by URL (with parameters intact) instead of clicking filters on the page.
- Wait for network idle or required selectors before scraping.

### 5. Validation
- After scraping, confirm that every row in the dataset matches **all** parameter-based filters (e.g., ratings ≥ `user_rating`, votes ≥ `num_votes`).
- If a mismatch occurs, re-check if a parameter was dropped or altered during pagination.

### 6. Safety & Consistency
- Do NOT hardcode parameter values unless given in the question or base URL.
- Do NOT guess parameters — only use documented or visible query params from the site.
- For each generated sourcing step, **output the final constructed URL** alongside the scraping code in the plan.
"""

    def url_pagination_prompt(self):
        return (
            "The provided URL contains paginated content. "
            "Plan the sourcing step to detect pagination links or API calls, "
            "iterate over all pages until no further results remain, "
            "and merge the extracted data into a single dataset."
        )

    def url_table_prompt(self):
        return """
    You are an expert data sourcing and preparation agent.

    Your task is to generate a detailed, step-by-step plan to extract and prepare structured data from an HTML source (URL) that contains tables and possibly other relevant HTML elements.

    Follow these **mandatory universal rules** for *all* websites:

    ---

    ### 1. Table Detection & Selection
    - Use `pandas.read_html` to extract all tables from the page.
    - Identify all tables relevant to the user’s questions — there may be more than one.
    - Do not assume the first table is the only one; consider all tables and their relationships.
    - If no relevant table is found, parse raw HTML elements (e.g., lists, `<div>` blocks, `<span>` text) using BeautifulSoup or lxml and PLaywright.

    ---

    ### 2. Multi-table Merging
    - If data required for a question is spread across multiple tables, determine a join key (e.g., player name, team name, ID, year) and merge them.
    - Normalize text (strip spaces, lowercase, remove accents) before joining.
    - Use inner joins unless specified otherwise.
    - If a join key is not unique, deduplicate by the most relevant context (e.g., latest year, matching category).

    ---

    ### 3. Raw HTML Parsing (Fallback)
    - If information is outside tables (e.g., bullet lists, paragraphs, sidebars), scrape it directly.
    - Extract specific attributes, links, or metadata if relevant.
    - Combine HTML-parsed data with table data where necessary.

    ---

    ### 4. Data Cleaning & Transformation
    - Remove footnotes, citations, and annotations (e.g., “[1]”, “(a)”, “†”) from all text fields.
    - Parse date/year fields into integers or datetime objects.
    - Ensure numeric fields are properly cast to `int` or `float`.
    - Trim whitespace and unify case in text columns.
    - Handle missing values appropriately (drop or fill based on context).
    - Sort data consistently if relevant to analysis.

    ---

    ### 5. Validation
    - Confirm that all columns required for the user’s questions exist after cleaning.
    - Verify that joins between tables are correct and yield complete datasets.
    - Ensure that all filtered datasets still have rows before analysis.

    ---

    ### 6. Output
    - Produce a clean, merged, and validated DataFrame ready for analysis.
    - Include all intermediate transformations in the plan for reproducibility.

    ---

    **Important:**  
    Your plan must include all necessary sourcing, cleaning, transformation, and merging steps explicitly.  
    Never assume that the question can be answered from a single table without verifying.
    """

    def url_api_prompt(self):
        return (
            "The provided URL is an API endpoint. "
            "Plan the sourcing step to issue API requests with the correct method, headers, authentication, and query parameters. "
            "If pagination or filtering applies, include logic to fetch all relevant data."
        )

    def url_text_only_prompt(self):
        return (
            "The provided URL contains structured text content without HTML tables or API endpoints. "
            "Plan the sourcing step to extract the relevant textual information using an HTML parser such as BeautifulSoup, "
            "applying targeted selectors to isolate required data."
        )

    def html_instructions(self):
        return """
**MANDATORY IMPORTANT INSTRUCTIONS THAT SHOULD BE PRESENT FOR URL SOURCING (HTML)**

1) special_instrcutions MUST BE A JSON OBJECT (not a list) filled with actual extracted values:
{
    "unique_table_identifier": "<identifier>",
    "code_to_get_table" : "<>",
    "raw_html_snippet": "<table class='...' ...>",
    "columns": ["ColA","ColB","..."],
    "noisy_values_per_column": {"ColA": ["$2,345[12]","3,210†"], "...": ["..."]},
    "row_samples": [ {"ColA":"...","ColB":"..."}, {"ColA":"...","ColB":"..."} ]
}

- [COMPULSORY] In raw_html snippet include at bare minimum a opening table tag and a row tag 
- [COMPULSORY] If any noisy values are there in data_files always include them as they are and add specific cleaning operations as required 
- [COMPULSORY] add the cleaning of columns in transformations using regex expressions for numerical data types always to handle noisy values efficiently
- [COMPULSORY] **Include** a `"code_to_get_table"` field in `special_instrcutions` with the exact Python (e.g., BeautifulSoup or pandas.read_html) code required to locate and extract the identified table from the HTML.

2) Provide (when available) a compact url_metadata.html_metadata.ranked_tables (top-5) for fallback:
"url_metadata": {
    "url": "<the url>",
    "html_metadata": {
    "title": "<page title>",
    "tables_total": <int>,
    "ranked_tables": [
        {
        "idx": <int>,
        "score": <0..1>,
        "caption": "<caption or nearby heading>",
        "selector": "table.wikitable:nth-of-type(k)",
        "opening_tag": "<table class='wikitable' ...>",
        "headers": ["..."],
        "noisy_values": {"Col": ["examples"]},
        "row_samples": [{"Col":"val", "...":"..."}]
        }
    ]
    }
}

3) Cleaning / typing rules (apply during extraction and later in code):
- Strip bracketed refs [..] and <sup>..</sup>; collapse spaces; trim.
- Numeric parse is clean-then-cast: keep digits/.,-()% and parentheses, then cast.
- For integers from noisy strings, use regex (e.g., re.search(r'\\d+', value)) before int().
- Dates: parse with pandas.to_datetime(..., errors='coerce') or dateutil; be timezone-aware if timestamps exist.
- If tables repeat headers across pages/sections, reuse the header globally.

4) Selection / pagination:
- Choose the table whose caption/near headings best match the question intent; include its CSS selector.
- If data spans multiple pages/“next” links, fetch only the minimal range needed to answer, then merge.
- If the stage page lacks details (e.g., MVP, roster nationalities), follow the team/match links present on the page.

        """

#     def html_instructions(self):
#         return """
# **MANDATORY IMPORTANT INSTRUCTIONS FOR URL SOURCING (HTML) — PLANNING ONLY (NOT EXECUTION)**

# 0) Discovery (planning, not code):
# - Assume the page may change; **do not** rely on fixed table index. Plan must select tables **by headers/caption match**, not position.
# - State that fetching should use a desktop User-Agent; if key tables are missing/collapsed, execution may use **sync Playwright** to render before parsing.

# 1) special_instrcutions MUST BE A JSON OBJECT (not a list) with **actual extracted samples** (no placeholders):
# {
#   "unique_table_identifier": "<best CSS selector like table.wikitable:nth-of-type(k) AFTER confirming headers>",
#   "raw_html_snippet": "<table class='...' ...>\\n<tr>...</tr>",
#   "columns": ["ColA","ColB","..."],                            // cleaned header names
#   "noisy_values_per_column": {"ColA": ["$2,345[12]","3,210†"], "ColB": ["23RK"]},  // up to 3 per column, raw strings only
#   "row_samples": [ {"ColA":"...","ColB":"..."}, {"ColA":"...","ColB":"..."} ]      // 2–3 rows
# }
# - [COMPULSORY] `raw_html_snippet` must include the opening <table ...> and at least one <tr>.
# - [COMPULSORY] If noisy values exist, include them exactly as found and specify regex cleaning steps under `transformations`.

# 2) Provide (when available) compact url_metadata.html_metadata.ranked_tables (top-5) for fallback:
# "url_metadata": {
#   "url": "<the url>",
#   "html_metadata": {
#     "title": "<page title>",
#     "tables_total": <int>,
#     "ranked_tables": [
#       {
#         "idx": <int>,                         // source index if known; not relied upon
#         "score": <0..1>,                      // heuristic header/caption match score
#         "caption": "<caption or nearby heading>",
#         "selector": "table.wikitable:nth-of-type(k)",
#         "opening_tag": "<table class='wikitable' ...>",
#         "headers": ["..."],
#         "noisy_values": {"Col": ["examples"]},
#         "row_samples": [{"Col":"val","...":"..."}]
#       }
#     ]
#   }
# }

# 3) Cleaning / typing rules to be listed in `transformations` (execution will implement):
# - Universal: strip bracketed refs `\\[.*?\\]`, remove `<sup>.*?</sup>`, normalize dashes (`−–—` → `-`), collapse spaces, trim.
# - Numeric parse is **clean-then-cast**: keep digits/.,-()% and parentheses, then `to_numeric(errors="coerce")`.
# - Integers from noisy strings: extract with regex `re.search(r'\\d+', ...)` or `str.extract(r'(\\d+)')` before cast.
# - Currency: remove `[$€£¥₹]` and commas before cast.
# - Years: extract first 4 digits with `str.extract(r'(\\d{4})')` then cast.
# - Rank/Peak-like: extract first integer token; cast to nullable integer in execution (no forced drop here).

# 4) Selection / pagination (planning directives):
# - Choose the table whose **headers include the required fields** for the questions (e.g., `Established`, `Title`, `Record-setting gross`), not by index.
# - If multiple candidates: prefer (a) more required columns matched, (b) more rows, (c) more relevant caption/heading/class.
# - If data spans multiple pages (“next” links, infinite scroll) or details live on linked team/match pages, **plan** to follow only the minimal set of pages/links needed; merge at analysis time.

# 5) Validation & NA policy (planning, to avoid wrong answers):
# - `validation` must confirm required columns **after header flattening** (e.g., `'Established','Title','Record-setting gross'`). Do **not** require unused columns (e.g., `Ref`) if questions don’t need them.
# - Explicitly state: **no global row drops** or imputations in execution. Each question will filter rows only on the columns it actually needs (e.g., correlation uses pairwise non-null of the two columns only).

# 6) Question mapping (make it explicit in plan):
# - For counting/sorting by year/gross: require only `Year/Established` and `Gross` be present/non-null.
# - For “max gross → title”: require non-null `Record-setting gross`; return the corresponding `Title`.
# - For correlations: use pairwise non-null of the two variables only.
# - For plots: note size constraint (<100k chars for base64 PNG); execution may reduce DPI/size if needed.

# 7) Deliverables in plan:
# - `data_sourcing.instructions`: say “header-based selection (not index)”, UA, optional sync Playwright if missing.
# - `validation`: list only columns truly required by the questions.
# - `transformations`: include concrete regex steps for the identified noisy columns (see §3).
# - `special_instrcutions`: populated JSON as in §1.
# - `url_metadata.html_metadata.ranked_tables`: provide up to 5 candidates as in §2.

# """




#     def html_instructions(self):
#         return """
# """
# """
# ** IMPORTANT AND MANDATORY INSTRUCTIONS TO BE FOLLOWED FOR ANY KIND OF URLS**

# - In the special instructions include
#     - Table : 
#         * If tables are required to be extracted table 
#         * Specify all the relevant tables name which will be used for retrieval
#         * the correct table index number from the url (it must be an integer scan the html to get it if required)
#         * the unique attribute for that particular table from which the table can be sourced , if class give the entire class attribute
#         * columns and for each columns the desired data type and scan all entries to get the 2 - 3 examples of noisy values present
#             -if the column conatins all numerical values but has one value like 23RK, give a list of these values which will be a trouble while typecasting from retrieval of html content through playwright
#         * raw html table sample that we get when sourcing 
# """

# def url_new_short_instructions(self):
#     return """


# Additional Instructions for url scraping, parquet file scraping, or any other source scraping:

# **DATA SOURCING EFFICIENCY:**
# 1. **Column Selection**: Always specify ONLY the required columns in your queries. Use SELECT statements with explicit column names instead of SELECT *.
# 2. **Filtering at Source**: Apply WHERE clauses directly in your data loading queries to filter data at the source level, avoiding loading unnecessary rows.
# 3. **Partitioning Awareness**: If data is partitioned (like year=2025/court=xyz), use partition predicates in your queries to scan only relevant partitions.
# 4. **Memory Management**: For large datasets, consider using chunked reading or streaming approaches rather than loading entire datasets into memory.

# **DUCKDB QUERY OPTIMIZATION:**
# 1. **Use DuckDB for Large Data**: When dealing with parquet files or large datasets, prefer DuckDB queries over pandas operations for better performance.
# 2. **Predicate Pushdown**: Structure queries to push filters down to the scan level: 
# ```sql
# SELECT required_columns FROM source WHERE conditions ORDER BY column LIMIT n
# ```
# 3. **Aggregations**: Perform grouping and aggregations at the SQL level rather than in Python when possible.
# 4. **Connection Management**: Reuse DuckDB connections and consider using persistent connections for multiple operations.

# **CODE VALIDATION AND REUSE:**
# 1. **Existing Code Integration**: If code snippets are provided in questions, validate their correctness but preserve the core logic.
# 2. **Column Mapping**: Verify that column names in existing code match the actual data schema and provide mapping if needed.
# 3. **Syntax Adaptation**: Adapt existing code to work with the specific data source format (CSV, parquet, JSON, etc.).

# **DATA INFERENCE AND ANALYSIS:**
# 1. **Derived Insights**: Look beyond direct column values to infer meaningful insights. Examples:
# - Population density from population/area ratios
# - Growth rates from temporal comparisons
# - Delay calculations from date differences
# - Categorical distributions and patterns
# 2. **Statistical Relationships**: Identify opportunities for correlation analysis, regression, trend analysis, and comparative statistics.
# 3. **Data Quality Assessment**: Include validation steps to check for missing values, outliers, and data consistency.
# 4. **Temporal Analysis**: For time-series data, consider seasonality, trends, and period-over-period comparisons.

# **COMPLEX QUERY PATTERNS:**
# 1. **Multi-table Operations**: When joining multiple sources, specify the join strategy and key columns clearly.
# 2. **Window Functions**: Use window functions for running totals, rankings, and period comparisons.
# 3. **Conditional Logic**: Implement CASE statements for categorization and conditional aggregations.

# **JSON OUTPUT FORMATTING:**
# 1. **Structured Responses**: Always return results in the specified JSON format with exact key names as requested.
# 2. **Data Type Consistency**: Ensure numeric results are returned as numbers, not strings, unless specifically requested otherwise.
# 3. **Precision Control**: For floating-point numbers, specify appropriate decimal precision (typically 2-4 decimal places).
# 4. **Visualization Encoding**: For charts/plots, encode as base64 data URIs in the specified format (PNG, WEBP, etc.) with size constraints.

# **EXAMPLE ANALYSIS PATTERNS:**
# 1. **Court Case Analysis**: 
# - Time-based aggregations: cases per year/month/court
# - Duration calculations: registration_date to decision_date differences
# - Disposal pattern analysis: outcomes by court/judge/case type
# - Trend analysis: regression slopes for temporal patterns

# 2. **Performance Metrics**:
# - Ranking operations: courts by case volume, efficiency metrics
# - Comparative analysis: court performance benchmarking
# - Statistical correlations: relationship between variables

# **VALIDATION AND ERROR HANDLING:**
# 1. **Data Validation**: Include checks for data availability, expected ranges, and data types.
# 2. **Fallback Strategies**: Provide alternative approaches if primary data sources are unavailable.
# 3. **Error Reporting**: Include meaningful error messages and debugging information.

# **SPECIFIC OPTIMIZATIONS FOR LARGE DATASETS:**
# 1. **Sampling**: For exploratory analysis on very large datasets, consider representative sampling.
# 2. **Indexing**: Leverage existing indexes and partitioning schemes.
# 3. **Batch Processing**: Break large operations into manageable chunks.
# 4. **Result Caching**: Store intermediate results to avoid recomputation.

# **ANSWER FORMAT REQUIREMENTS:**
# 1. **JSON Structure**: Always return answers in the exact JSON format requested in the question.
# 2. **Key Naming**: Use the exact question text as JSON keys unless otherwise specified.
# 3. **Value Types**: Return appropriate data types (strings for text, numbers for numeric results, base64 strings for images).
# 4. **Completeness**: Ensure all requested questions are answered in the JSON response.

# **VISUALIZATION GUIDELINES:**
# 1. **Chart Selection**: Choose appropriate chart types for the data (scatter plots for correlations, bar charts for comparisons, line charts for trends).
# 2. **Base64 Encoding**: For image outputs, use efficient encoding and compression to stay within character limits.
# 3. **Color and Styling**: Use clear, professional styling with appropriate colors and labels.
# 4. **Size Optimization**: Optimize image size while maintaining readability and staying within the specified character limits.
# """